import sys, torch, numpy as np
sys.path.insert(0, 'stage1_segmentation')
sys.path.insert(0, 'stage2_landmarks')
sys.path.insert(0, 'extensions/tgnet_ops')
sys.path.insert(0, 'extensions/teethland_ops')
sys.path.insert(0, '.')

from pipeline.combined_pipeline import CombinedDentalPipeline
from pipeline.data_bridge import get_scale_factor
import gen_utils as gu

Z_SCORE_STD = 17.3281

pipeline = CombinedDentalPipeline(
    tgnet_fps_ckpt='checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5',
    tgnet_bdl_ckpt='checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5',
    landmark_ckpt='checkpoints/Teethland-checkpoints/landmarks_full.ckpt',
    device='cuda',
)
obj_path = 'data/teeth3ds_sample/01F4JV8X/01F4JV8X_upper.obj'

original_verts = gu.read_txt_obj_ls(obj_path, ret_mesh=False, use_tri_mesh=False)[0]
orig_xyz_mm   = original_verts[:, :3]
orig_normals  = original_verts[:, 3:]
scale_factor  = get_scale_factor(orig_xyz_mm)
jaw_mean      = orig_xyz_mm.mean(axis=0)
jaw_norm      = (orig_xyz_mm - jaw_mean) / Z_SCORE_STD
min_y_c       = float(orig_xyz_mm[:, 1].min()) - float(jaw_mean[1])
print('Scale factor:', scale_factor)
print('jaw_mean:', jaw_mean)
print('jaw_norm range:', jaw_norm.min(axis=0), '-', jaw_norm.max(axis=0))

seg_result = pipeline.seg_pipeline.run(obj_path)
sampled_xyz  = seg_result['sampled_xyz']
labels = np.array(seg_result['sampled_sem_labels'])
unique_fdi = sorted(set(int(l) for l in labels if l != 0))
print('Unique FDI labels:', unique_fdi[:8])

fdi_label = unique_fdi[0]
tooth_mask = labels == fdi_label
tooth_norm_xyz  = sampled_xyz[tooth_mask]
centroid_tgnet  = tooth_norm_xyz.mean(axis=0)
centroid_norm   = ((centroid_tgnet + 0.8) * scale_factor + min_y_c) / Z_SCORE_STD
print(f'FDI {fdi_label}: {tooth_mask.sum()} sampled points')
print('centroid_tgnet:', centroid_tgnet)
print('centroid_norm:', centroid_norm)

from sklearn.neighbors import KDTree as _KDTree
norm_tree = _KDTree(jaw_norm, leaf_size=16)
crop_idxs = norm_tree.query(centroid_norm[None], k=12000, return_distance=False)[0]
crop_xyz_norm  = jaw_norm[crop_idxs]
crop_normals   = orig_normals[crop_idxs]
centroid_offsets = crop_xyz_norm - centroid_norm

crop_xyz_t   = torch.from_numpy(crop_xyz_norm).float().cuda()
crop_norm_t  = torch.from_numpy(crop_normals).float().cuda()
cent_off_t   = torch.from_numpy(centroid_offsets).float().cuda()
features     = torch.cat([crop_xyz_t, crop_norm_t, cent_off_t], dim=1)

from teethland import PointTensor as _PT
pt = _PT(
    coordinates=crop_xyz_t,
    features=features,
    batch_counts=torch.tensor([crop_xyz_t.shape[0]], device='cuda'),
)
print('C range (norm):', pt.C.min().item(), '-', pt.C.max().item())
print('F shape:', pt.F.shape, 'F range:', pt.F.min().item(), '-', pt.F.max().item())
print('F[0:3] range (xyz_norm):', pt.F[:,:3].min().item(), '-', pt.F[:,:3].max().item())
print('F[6:9] range (offsets):', pt.F[:,6:].min().item(), '-', pt.F[:,6:].max().item())

with torch.no_grad():
    seg_head, *lm_heads = pipeline.landmark_model(pt)

print('seg_head F:', seg_head.F.min().item(), '-', seg_head.F.max().item())
for i, h in enumerate(lm_heads):
    dist = h.F[:, 0]
    print(f'lm_head[{i}] dist: min={dist.min():.4f} max={dist.max():.4f} below_0.12={(dist<0.12).sum().item()}')
