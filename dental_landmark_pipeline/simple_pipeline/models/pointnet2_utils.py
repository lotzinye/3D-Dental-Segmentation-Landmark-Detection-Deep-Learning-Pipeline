"""
pointnet2_utils.py
------------------
Pure-PyTorch implementations of the core PointNet++ operations.

No custom CUDA compilation is required.  If you install torch-cluster
(a pre-built binary wheel — no compilation):

    pip install torch-cluster

the Farthest Point Sampling step will use its fast C++ kernel.
Otherwise a pure-PyTorch fallback is used (slower, but correct).

Public API
----------
farthest_point_sample(xyz, npoint)           -> (B, npoint)  long
index_points(points, idx)                    -> (B, *, C)    float
ball_query(radius, nsample, xyz, new_xyz)    -> (B, S, nsample) long
PointNetSetAbstraction                       nn.Module  (single-scale SA)
PointNetSetAbstractionMsg                    nn.Module  (multi-scale SA)
PointNetFeaturePropagation                   nn.Module  (FP / decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional fast FPS from torch-cluster (pre-built wheel, no compilation)
# ---------------------------------------------------------------------------
try:
    from torch_cluster import fps as _tc_fps

    def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Farthest Point Sampling using torch-cluster (fast).

        Selects `npoint` spatially spread-out points from each batch element.

        Args:
            xyz:    (B, N, 3) float32 — input point positions
            npoint: number of centroids to sample

        Returns:
            idx: (B, npoint) long — indices into the N dimension
        """
        B, N, _ = xyz.shape
        ratio = npoint / N
        results = []
        for b in range(B):
            idx = _tc_fps(xyz[b], ratio=ratio, random_start=True)
            # torch-cluster may return npoint±1 due to rounding; clip to exact
            results.append(idx[:npoint])
        return torch.stack(results, dim=0)

except ImportError:

    def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Farthest Point Sampling — pure-PyTorch fallback.

        Iteratively selects the point farthest from all already-selected
        centroids.  O(N × npoint) but fully vectorised inner loop.

        Args:
            xyz:    (B, N, 3) float32
            npoint: number of centroids

        Returns:
            idx: (B, npoint) long
        """
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        dist      = torch.full((B, N), 1e10, device=device)
        farthest  = torch.randint(0, N, (B,), device=device)
        batch_idx = torch.arange(B, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_idx, farthest].unsqueeze(1)   # (B, 1, 3)
            d = ((xyz - centroid) ** 2).sum(dim=-1)             # (B, N)
            dist = torch.minimum(dist, d)
            farthest = dist.argmax(dim=-1)                      # (B,)

        return centroids


# ---------------------------------------------------------------------------
# Helper: gather points by index
# ---------------------------------------------------------------------------

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather point features at arbitrary indices.

    Args:
        points: (B, N, C)
        idx:    (B, *) long — any shape of indices

    Returns:
        out: (B, *, C) — features at the requested indices
    """
    B, N, C = points.shape
    flat_idx = idx.reshape(B, -1)                           # (B, M)
    out = torch.gather(
        points,
        dim=1,
        index=flat_idx.unsqueeze(-1).expand(-1, -1, C),
    )                                                       # (B, M, C)
    return out.reshape(list(idx.shape) + [C])


# ---------------------------------------------------------------------------
# Ball query
# ---------------------------------------------------------------------------

def ball_query(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    chunk: int = 256,
) -> torch.Tensor:
    """
    For each centroid in `new_xyz`, find the `nsample` nearest points in
    `xyz` that lie within `radius`.  If fewer than `nsample` points fall
    inside the ball, the closest in-ball point is repeated (standard
    PointNet++ behaviour).

    Processes centroids in chunks of `chunk` to keep peak memory bounded
    (avoids constructing the full (B, S, N) distance matrix at once).

    Args:
        radius:  ball radius
        nsample: max neighbours per centroid
        xyz:     (B, N, 3) — source point cloud
        new_xyz: (B, S, 3) — query centroid positions
        chunk:   centroid batch size for chunked distance computation

    Returns:
        idx: (B, S, nsample) long — indices into N for each centroid
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)

    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        q  = new_xyz[:, s0:s1, :]           # (B, cs, 3)
        d  = torch.cdist(q, xyz)            # (B, cs, N)  — Euclidean distances

        # Push out-of-radius distances to infinity so they sort last.
        d_masked = d.clone()
        d_masked[d >= radius] = 1e9

        # argsort ascending: first nsample entries are nearest in-ball points
        sorted_idx = d_masked.argsort(dim=-1)[:, :, :nsample]  # (B, cs, nsample)

        # For centroids with < nsample valid neighbours, fill gaps with index 0
        # of the sorted result (= closest valid point).
        n_valid = (d < radius).sum(dim=-1, keepdim=True)       # (B, cs, 1)
        slot    = torch.arange(nsample, device=device).view(1, 1, nsample)
        fill    = sorted_idx[:, :, :1].expand_as(sorted_idx)
        sorted_idx = torch.where(slot < n_valid, sorted_idx, fill)

        idx[:, s0:s1, :] = sorted_idx

    return idx


# ---------------------------------------------------------------------------
# Set Abstraction — single-scale
# ---------------------------------------------------------------------------

class PointNetSetAbstraction(nn.Module):
    """
    Single-Scale Set Abstraction (SA) layer.

    1. Sample `npoint` centroids via FPS.
    2. Group `nsample` neighbours within `radius` around each centroid.
    3. Apply a shared MLP (Conv2d) to each (centroid, neighbour) pair.
    4. Max-pool across neighbours → one feature vector per centroid.

    Args:
        npoint:    number of centroids (None → group_all mode)
        radius:    ball radius
        nsample:   neighbours per centroid
        in_channel: number of input feature channels (NOT counting xyz)
        mlp:       list of output channel widths for the shared MLP
        group_all: if True, skip FPS and group every point (global layer)
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint    = npoint
        self.radius    = radius
        self.nsample   = nsample
        self.group_all = group_all

        layers, last = [], in_channel + 3   # +3 for relative xyz
        for out_c in mlp:
            layers += [
                nn.Conv2d(last, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            last = out_c
        self.mlp_convs = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor,
    ):
        """
        Args:
            xyz:    (B, N, 3)  — point positions
            points: (B, C, N)  — point features (channels-first), or None

        Returns:
            new_xyz:    (B, S, 3)    — centroid positions
            new_points: (B, C', S)   — abstracted features (channels-first)
        """
        # Convert features to channels-last for grouping
        pts_t = points.permute(0, 2, 1) if points is not None else None  # (B,N,C)

        if self.group_all:
            new_xyz = torch.zeros(B := xyz.shape[0], 1, 3, device=xyz.device)
            grp_xyz = xyz.unsqueeze(1)                                  # (B,1,N,3)
            grp_pts = (
                torch.cat([grp_xyz, pts_t.unsqueeze(1)], dim=-1)
                if pts_t is not None else grp_xyz
            )
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)                        # (B,S,3)
            grp_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            grp_xyz = index_points(xyz, grp_idx) - new_xyz.unsqueeze(2) # (B,S,ns,3)
            if pts_t is not None:
                grp_pts = torch.cat(
                    [grp_xyz, index_points(pts_t, grp_idx)], dim=-1
                )
            else:
                grp_pts = grp_xyz

        # (B, S, ns, C) → (B, C, S, ns) for Conv2d
        grp_pts = grp_pts.permute(0, 3, 1, 2)
        grp_pts = self.mlp_convs(grp_pts)          # (B, C', S, ns)
        new_points = grp_pts.max(dim=-1)[0]         # (B, C', S)
        return new_xyz, new_points


# ---------------------------------------------------------------------------
# Set Abstraction — multi-scale grouping (MSG)
# ---------------------------------------------------------------------------

class PointNetSetAbstractionMsg(nn.Module):
    """
    Multi-Scale Grouping (MSG) Set Abstraction layer.

    Runs ball queries at multiple radii simultaneously and concatenates the
    resulting feature vectors.  Captures both fine local geometry (small
    radius) and broader context (large radius) from a single layer.

    Args:
        npoint:       number of centroids
        radius_list:  list of radii, one per scale
        nsample_list: list of max neighbours, one per scale
        in_channel:   raw feature channels at the input (NOT counting xyz)
        mlp_list:     list of MLP channel specs, one list per scale
    """

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint       = npoint
        self.radius_list  = radius_list
        self.nsample_list = nsample_list

        # One independent MLP per scale
        self.scale_mlps = nn.ModuleList()
        for mlp in mlp_list:
            layers, last = [], in_channel + 3
            for out_c in mlp:
                layers += [
                    nn.Conv2d(last, out_c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
                last = out_c
            self.scale_mlps.append(nn.Sequential(*layers))

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        """
        Args:
            xyz:    (B, N, 3)
            points: (B, C, N) or None

        Returns:
            new_xyz:    (B, S, 3)
            new_points: (B, sum(C'_i), S)   — all scales concatenated
        """
        pts_t   = points.permute(0, 2, 1) if points is not None else None
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)            # (B, S, 3)

        scale_feats = []
        for r, ns, mlp in zip(self.radius_list, self.nsample_list, self.scale_mlps):
            grp_idx = ball_query(r, ns, xyz, new_xyz)   # (B, S, ns)
            grp_xyz = index_points(xyz, grp_idx) - new_xyz.unsqueeze(2)

            if pts_t is not None:
                grp_pts = torch.cat(
                    [grp_xyz, index_points(pts_t, grp_idx)], dim=-1
                )
            else:
                grp_pts = grp_xyz

            grp_pts = grp_pts.permute(0, 3, 1, 2)      # (B, C, S, ns)
            grp_pts = mlp(grp_pts)                      # (B, C', S, ns)
            scale_feats.append(grp_pts.max(dim=-1)[0])  # (B, C', S)

        return new_xyz, torch.cat(scale_feats, dim=1)   # (B, sum_C', S)


# ---------------------------------------------------------------------------
# Feature Propagation — decoder / upsample
# ---------------------------------------------------------------------------

class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation (FP) layer.

    Upsamples coarse features (S points) back to a finer resolution (N points)
    using inverse-distance-weighted interpolation among the 3 nearest coarse
    neighbours, then concatenates the skip-connection features from the
    corresponding SA layer and refines with a 1-D shared MLP.

    Args:
        in_channel: total input channels (skip features + upsampled features)
        mlp:        list of output channel widths for the shared 1-D MLP
    """

    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        layers, last = [], in_channel
        for out_c in mlp:
            layers += [
                nn.Conv1d(last, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
            ]
            last = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz1:    (B, N, 3)  — target (finer) positions
            xyz2:    (B, S, 3)  — source (coarser) positions  [None → global]
            points1: (B, C1, N) — skip-connection features     [None → skip]
            points2: (B, C2, S) — features to upsample

        Returns:
            (B, C_out, N)
        """
        if xyz2 is None:
            # Global feature: broadcast to every point
            interpolated = points2.expand(-1, -1, xyz1.shape[1])
        else:
            B, N, _ = xyz1.shape
            dists = torch.cdist(xyz1, xyz2)                         # (B, N, S)
            k     = min(3, xyz2.shape[1])
            dists, idx = dists.topk(k=k, dim=-1, largest=False)    # (B, N, k)
            dists = torch.clamp(dists, min=1e-10)
            weights = (1.0 / dists)
            weights = weights / weights.sum(dim=-1, keepdim=True)   # (B, N, k)

            pts2_t     = points2.permute(0, 2, 1)                   # (B, S, C)
            nbr_feats  = index_points(pts2_t, idx)                  # (B, N, k, C)
            interpolated = (weights.unsqueeze(-1) * nbr_feats).sum(2)
            interpolated = interpolated.permute(0, 2, 1)            # (B, C, N)

        out = (
            torch.cat([points1, interpolated], dim=1)
            if points1 is not None else interpolated
        )
        return self.mlp(out)
