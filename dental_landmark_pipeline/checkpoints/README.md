# Checkpoints

All required model weights are already present in this folder.

## Actual file locations

| File | Path |
|------|------|
| TGNet FPS (Stage 1) | `CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5` |
| TGNet BDL (Stage 2) | `CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5` |
| 3DTeethLand LandmarkNet | `Teethland-checkpoints/landmarks_full.ckpt` |

> The TGNet files use a .h5 extension but are standard PyTorch checkpoints (ZIP format).
> torch.load() reads them correctly.

## Default paths

The pipeline defaults are already set to the paths above. To override:

    python run_pipeline.py scan.obj ^
        --fps-ckpt  "checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_fps.h5" ^
        --bdl-ckpt  "checkpoints/CGIP_TGN_checkpoints/ckpts(new)/tgnet_bdl.h5" ^
        --lm-ckpt   checkpoints/Teethland-checkpoints/landmarks_full.ckpt
