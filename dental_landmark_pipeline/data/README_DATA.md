# Dataset Setup

## Required datasets

| Dataset | Purpose | Download |
|---------|---------|---------|
| Teeth3DS (7 parts) | 3D jaw scans + per-vertex FDI labels | https://osf.io/xctdy/ |
| 3DTeethLand annotations | Anatomical landmark coordinates | https://osf.io/um96h/ |

---

## How to organise the data

### Step 1 — Extract Teeth3DS (7 parts)

Extract all 7 zip files. Each zip contains `upper/` and `lower/` subfolders.
Merge them all into two combined folders inside `data/`:

```
data/
├── upper/
│   ├── {PATIENT_ID}/
│   │   ├── {ID}_upper.obj
│   │   └── {ID}_upper.json
│   └── ...
└── lower/
    ├── {PATIENT_ID}/
    │   ├── {ID}_lower.obj
    │   └── {ID}_lower.json
    └── ...
```

When you extract all 7 parts, the patient folders accumulate in `upper/` and `lower/` —
just merge them (Windows will ask "merge" or "replace"; choose **merge**).

### Step 2 — Extract 3DTeethLand landmark annotations

Extract both landmark zips into their own named folders:

```
data/
├── 3DTeethLand_landmarks_train/
│   ├── upper/
│   │   └── {PATIENT_ID}/{ID}_upper__kpt.json   ← note: double underscore
│   └── lower/
│       └── {PATIENT_ID}/{ID}_lower__kpt.json
└── 3DTeethLand_landmarks_test/
    ├── upper/
    │   └── {PATIENT_ID}/{ID}_upper__kpt.json
    └── lower/
        └── {PATIENT_ID}/{ID}_lower__kpt.json
```

> **Important**: landmark files use a **double underscore** before `kpt`:
> `{ID}_upper__kpt.json` — this is how the dataset is distributed.

---

## Full layout after setup

```
data/
├── upper/                               ← Teeth3DS scans (all 7 parts combined)
├── lower/                               ← Teeth3DS scans (all 7 parts combined)
├── 3DTeethLand_landmarks_train/         ← landmark annotations, train split
├── 3DTeethLand_landmarks_test/          ← landmark annotations, test split
└── teeth3ds_sample/                     ← 1 sample patient (already included)
```

---

## Validate the dataset

After extracting, run:

```bat
python data/prepare_dataset.py
python data/prepare_dataset.py --verbose   # show per-file details
```

This prints a summary: scans found, how many have landmark annotations, any errors.

---

## JSON formats

### Segmentation JSON (`*_upper.json` / `*_lower.json`)

```json
{
    "id_patient": "6X24ILNE",
    "jaw": "upper",
    "labels":    [0, 0, 44, 33, 34, ...],
    "instances": [0, 0, 10,  2, 12, ...]
}
```

- `labels[i]`: FDI tooth number for vertex `i` (0 = gingiva)
- `instances[i]`: instance ID for vertex `i` (0 = gingiva)

### Landmark JSON (`*__kpt.json`)

```json
{
    "version": "1.0",
    "description": "landmarks",
    "key": "01A6HAN6_upper.obj",
    "objects": [
        {
            "key": "uuid_0",
            "class": "Mesial",
            "coord": [3.13, -19.71, -78.12]
        }
    ]
}
```

Landmark classes: `Mesial`, `Distal`, `FacialPoint`, `OuterPoint`, `InnerPoint`, `Cusp`

---

## Running the pipeline on a scan

```bat
python run_pipeline.py data/upper/01F4JV8X/01F4JV8X_upper.obj
```

Or use the included sample:

```bat
python run_pipeline.py data/teeth3ds_sample/teeth3ds_sample/01F4JV8X/01F4JV8X_upper.obj
```
