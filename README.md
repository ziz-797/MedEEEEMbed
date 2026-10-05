# MedHEB: Benchmarking Medical Embeddings Across Heterogeneous Clinical Evidence

MedHEB is a medical embedding evaluation benchmark covering 2D images, 3D volumes, and text.

## Quick Start

```bash
# Install dependencies
pip install transformers>=4.57 torch datasets accelerate qwen-vl-utils

# Run evaluation
bash scripts/evaluation/mediem/eval_embedding.sh [model_path]
```

## Data

Follow [`data/README.md`](data/README.md) for dataset download and preprocessing instructions.

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
- `2D_Task.yaml` — 69 datasets (CLS ×16, I2I ×17, T2I ×18, VQA ×9, VG ×7, Report ×2)
- `3D_Task.yaml` — 31 datasets (CLS ×6, VQA ×3, I2T ×4, T2I ×5, I2I ×4, Cross-modal ×9)
- `Text_Task.yaml` — 7 datasets (T2T ×7)

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
