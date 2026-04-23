# MedEmbed - Medical Multimodal Embedding Evaluation

Medical embedding evaluation benchmark built on [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding), covering 2D images, 3D volumes, and text across 96 datasets.

## Quick Start

```bash
# Install dependencies
pip install transformers>=4.57.3 torch datasets accelerate qwen-vl-utils

# Run evaluation
bash scripts/evaluation/mediem/eval_embedding.sh [model_path]
```

## Data Structure

Organize your data under a single root directory (`DATA_BASEDIR`):

```
DATA_BASEDIR/
│
├── 2D_Task/                                    # 2D task annotation JSONs
│   ├── APTOS/                                  # Classification / I2I / T2I
│   │   ├── APTOS_test.json
│   │   ├── APTOS_i2i_test.json
│   │   └── APTOS_t2i_test.json
│   ├── BloodMNIST/
│   │   ├── BloodMNIST_test.json
│   │   ├── BloodMNIST_i2i_test.json
│   │   └── BloodMNIST_t2i_test.json
│   ├── Brain-Tumor-MRI/
│   │   ├── Brain_Tumor_MRI_test.json
│   │   ├── Brain_Tumor_MRI_i2i_test.json
│   │   └── Brain_Tumor_MRI_t2i_test.json
│   ├── ChestMNIST/
│   │   ├── ChestMNIST_test.json
│   │   ├── ChestMNIST_i2i_test.json
│   │   └── ChestMNIST_t2i_test.json
│   ├── DermaMNIST/
│   │   ├── DermaMNIST_test.json
│   │   ├── DermaMNIST_i2i_test.json
│   │   └── DermaMNIST_t2i_test.json
│   ├── ISIC-2109/
│   │   ├── ISIC_2019_test.json
│   │   ├── ISIC_2019_i2i_test.json
│   │   └── ISIC_2019_t2i_test.json
│   ├── Kvasir/
│   │   ├── Kvasir_test.json
│   │   ├── Kvasir_i2i_test.json
│   │   └── Kvasir_t2i_test.json
│   ├── MIMIC-CXR-T/
│   │   ├── mimic-cxr-lt-test.json
│   │   ├── mimic-cxr-lt-i2i-test.json
│   │   └── mimic-cxr-lt-t2i-test.json
│   ├── OCT/
│   │   ├── OCT_test.json
│   │   ├── OCT_i2i_test.json
│   │   └── OCT_t2i_test.json
│   ├── OCTMNIST/
│   │   ├── OCTMNIST_test.json
│   │   ├── OCTMNIST_i2i_test.json
│   │   └── OCTMNIST_t2i_test.json
│   ├── OrganAMNIST/
│   │   ├── OrganAMNIST_test.json
│   │   ├── OrganAMNIST_i2i_test.json
│   │   └── OrganAMNIST_t2i_test.json
│   ├── OrganCMNIST/
│   │   ├── OrganCMNIST_test.json
│   │   ├── OrganCMNIST_i2i_test.json
│   │   └── OrganCMNIST_t2i_test.json
│   ├── OrganSMNIST/
│   │   ├── OrganSMNIST_test.json
│   │   ├── OrganSMNIST_i2i_test.json
│   │   └── OrganSMNIST_t2i_test.json
│   ├── PanNuke/
│   │   ├── pannuke_test.json
│   │   ├── pannuke_i2i_test.json
│   │   └── pannuke_t2i_test.json
│   ├── PathMNIST/
│   │   ├── PathMNIST_test.json
│   │   ├── PathMNIST_i2i_test.json
│   │   └── PathMNIST_t2i_test.json
│   ├── SD-198/
│   │   ├── SD-198_test.json
│   │   ├── SD-198_i2i_test.json
│   │   └── SD-198_t2i_test.json
│   ├── TissueMNIST/
│   │   ├── TissueMNIST_test.json
│   │   ├── TissueMNIST_i2i_test.json
│   │   └── TissueMNIST_t2i_test.json
│   ├── PathVQA/                                # VQA
│   │   └── Path-VQA_test.json
│   ├── RSNA-Bone/
│   │   └── Bone_test.json
│   ├── PMC/
│   │   └── PMC-VQA_test.json
│   ├── ROCO-VQA/
│   │   └── ROCO_test.json
│   ├── MedPIX/
│   │   └── MedPix_test.json
│   ├── RadImageNet-VQA/
│   │   └── RadLmageNet_test.json
│   ├── VQA-RAD/
│   │   └── VQA_test.json
│   ├── MIMIC-CXR-VQA/
│   │   └── mimic_test.json
│   ├── DeepLesion/                             # Visual Grounding
│   │   └── DeepLesion_test.json
│   ├── PanNuke_VG/
│   │   └── PanNuke_test.json
│   ├── UltrasoundNerve/
│   │   └── Ultrasound_test.json
│   ├── ChestImagrome/
│   │   └── Chest_imagenome_test.json
│   ├── Fundus/
│   │   └── fundus_test.json
│   ├── Gastrointestinal/
│   │   └── gastrointestinal_test.json
│   ├── SkinLesion/
│   │   └── skin_test.json
│   ├── VindrCXR/
│   │   └── VinDr-CXR_test.json
│   ├── VindrMammo/
│   │   └── VindrMammo_test.json
│   ├── Promise12/
│   │   └── promise12_test.json
│   ├── MIMIC-CXR-Report/                      # Report Generation
│   │   └── MIMIC-CXR_test.json
│   └── USData/
│       └── USData_test.json
│
├── 2D_Images/                                  # 2D image files
│   ├── APTOS/train_images/
│   ├── BloodMNIST/output_images/test/
│   ├── Brain-Tumor-MRI/datasets/Testing/
│   ├── ChestMNIST/output_images/test/
│   ├── DermaMNIST/output_images/test/
│   ├── ISIC-2019/test/ISIC_2019_Test_Input/
│   ├── kvasir-dataset/datasets/test/
│   ├── MIMIC_CXR/                              # with files/ symlink
│   ├── ChestImagrome/                          # gold_crop/ + p10..p19,files symlinks → ../MIMIC_CXR/
│   ├── OCT/datasets/test/
│   ├── OCTMNIST/output_images/test/
│   ├── OrganAMNIST/output_images/test/
│   ├── OrganCMNIST/output_images/test/
│   ├── OrganSMNIST/output_images/test/
│   ├── PanNuke/test_images/
│   ├── PathMNIST/output_images/test/
│   ├── SD-198/images/sd-198/images/
│   ├── TissueMNIST/output_images/test/
│   ├── PathVQA/test_images/
│   ├── RSNA-Bone/
│   ├── PMC/
│   ├── ROCO-VQA/
│   ├── MedPIX/
│   ├── RadImageNet-VQA/test/images/
│   ├── VQA-RAD/
│   ├── DeepLesion/
│   ├── PanNuke_VG/
│   ├── UltrasoundNerve/
│   ├── Fundus/
│   ├── Gastrointestinal/
│   ├── SkinLesion/
│   ├── VindrCXR/
│   ├── VindrMammo/
│   ├── Promise12/
│   └── USData/
│
├── 3D_Task/                                    # 3D task annotation JSONs
│   ├── CT_RATE/                                # CLS / VQA / I2T / T2I / I2I
│   │   ├── CT_RATE_test.json
│   │   ├── CT_RATE_vqa_test.json
│   │   ├── CT_RATE_i2t_test.json
│   │   ├── CT_RATE_t2i_test.json
│   │   └── CT_RATE_i2i_test.json
│   ├── ChirrMRI600/
│   │   ├── Chirr_test.json
│   │   ├── Chirr_i2i_test.json
│   │   └── Chirr_t2i_test.json
│   ├── MRNet/
│   │   ├── MRNet_test.json
│   │   ├── MRNet_i2i_test.json
│   │   └── MRNet_t2i_test.json
│   ├── NoduleMNIST/
│   │   └── Nodule_test.json
│   ├── Organ3dMNIST/
│   │   ├── Organ_test.json
│   │   ├── Organ_i2i_test.json
│   │   └── Organ_t2i_test.json
│   ├── SynapseMNIST/
│   │   └── Synapse_test.json
│   ├── BIMCV-R/
│   │   └── BIMCV-R_i2t_test.json
│   ├── RadGen_CT/
│   │   ├── RGCC_i2t_test.json
│   │   ├── RGCC_t2i_test.json
│   │   └── RGCC_test_VQA.json
│   ├── M3D/
│   │   ├── m3d_3dqa_test.json
│   │   ├── m3d_i2t_test.json
│   │   └── m3d_t2i_test.json
│   ├── MMDental/
│   │   └── MMDental_i2t_test.json
│   ├── BraTS2023/                              # Cross-modal (T1 <-> T2)
│   │   ├── bratsped_t1_to_t2_test.json
│   │   └── bratsped_t2_to_t1_test.json
│   ├── CirrMRI600_Cross/
│   │   ├── ChirrMRI_test_t1_to_t2.json
│   │   └── ChirrMRI_test_t2_to_t1.json
│   ├── HaN-Seg/
│   │   ├── hanseg_ct_to_mri_test.json
│   │   └── hanseg_mri_to_ct_test.json
│   └── SynthRAD/
│       ├── brain_ct_to_mri_test.json
│       ├── brain_mri_to_ct_test.json
│       ├── pelvis_ct_to_mri_test.json
│       └── pelvis_mri_to_ct_test.json
│
├── 3D_Images/                                  # 3D image/slice files
│   ├── CT_RATE/
│   │   ├── valid_fixed_sliced/
│   │   └── valid_fixed_drr/
│   ├── ChirrMRI600/Cirrhosis_T1_slide/
│   ├── MRNet/valid_images/
│   ├── NoduleMNIST/test_slides/
│   ├── Organ3dMNIST/test_slides/
│   ├── SynapseMNIST/test_slides/
│   ├── BIMCV-R/ct_slice/
│   ├── RadGen_CT/valid_preprocessed_sliced/
│   ├── M3D/
│   ├── MMDental/cbct_png/
│   ├── BraTS2023/
│   ├── CirrMRI600_Cross/
│   ├── HaN-Seg/
│   └── SynthRAD/images/
│
└── Text_Task/                                  # Text-only task JSONs
    ├── MedMCQA/
    │   └── medmcqa_test.json
    ├── MIMIC_f2i/
    │   └── MIMIC_f2i_test.json
    ├── PubMedQA/
    │   └── PubMedQA_test.json
    ├── MedicalQARetrieval/
    │   └── MedicalQARetrieval_test.json
    ├── PublicHealthQA/
    │   └── PublicHealthQA_test.json
    └── MMDental/
        └── MMDental_t2t_test.json
```

### Setup Notes

**ChestImagenome** mixes MIMIC-CXR originals (`qry_img_path`: `p19/...`) with bbox crops (`tgt_img_path`: `gold_crop/...`). The crops live under `2D_Images/ChestImagrome/`, so the MIMIC patient dirs are symlinked in:

```bash
cd 2D_Images/ChestImagrome
for d in p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 files; do
    ln -sfn ../MIMIC_CXR/$d $d
done
```

With these symlinks, the YAML uses `image_root: 2D_Images/ChestImagrome` to resolve both query and target paths.

## JSON Data Format

All task JSONs follow a unified format:

**Image/Volume to Text** (classification, VQA, report):
```json
{
  "qry_inst": "Represent the given image with the following question:",
  "qry_text": "What is the severity level?",
  "qry_img_path": "image.png",
  "tgt_text": ["label_1", "label_2", "..."]
}
```

For 3D volumes, `qry_img_path` is a list of slice paths, with `qry_modality: "video"`:
```json
{
  "qry_inst": "Represent the given CT series:",
  "qry_text": "What abnormalities are present?",
  "qry_img_path": ["slice_000.png", "slice_001.png", "..."],
  "qry_modality": "video",
  "tgt_text": ["finding_1", "finding_2", "..."]
}
```

**Image/Volume to Image/Volume** (retrieval):
```json
{
  "qry_inst": "Find similar images:",
  "qry_text": "",
  "qry_img_path": "query.png",
  "tgt_text": ["", "", "..."],
  "tgt_img_path": ["cand_1.png", "cand_2.png", "..."]
}
```

**Text to Text**:
```json
{
  "qry_inst": "Find the answer:",
  "qry_text": "What causes this condition?",
  "tgt_text": ["answer_1", "answer_2", "..."]
}
```

## Evaluation Config

YAML configs in `scripts/evaluation/mediem/`:
- `2D_Task.yaml` — 60 datasets (CLS ×14, I2I ×15, T2I ×15, VQA ×7, VG ×7, Report ×2)
- `3D_Task.yaml` — 30 datasets (CLS ×6, VQA ×3, I2T ×3, T2I ×5, I2I ×4, Cross-modal ×9)
- `Text_Task.yaml` — 6 datasets (T2T ×6)

Set `DATA_BASEDIR` in `eval_embedding.sh` to your data root.

## Dataset Parsers

| Parser | File | Description |
|--------|------|-------------|
| `image_cls` / `image_qa` | `image_i2t.py` | 2D image(s) + text query → text candidates |
| `image_i2i` / `image_i2i_vg` | `image_i2i.py` | 2D image query → image candidates (aliases; VG tasks use the `_vg` name for readability) |
| `image_t2i` | `image_t2i.py` | Text query → 2D image candidates |
| `text_qa` | `text_t2t.py` | Text query → text candidates |
| `volume_i2t` | `volume_i2t.py` | 3D volume query → text candidates |
| `volume_t2i` | `volume_t2i.py` | Text query → 3D volume candidates |
| `volume_i2i` | `volume_i2i.py` | Volume/image query → volume candidates |

## Acknowledgments

Built on [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) and evaluation framework adapted from [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec).
