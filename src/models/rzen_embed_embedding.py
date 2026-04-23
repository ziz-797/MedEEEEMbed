"""Adapter that wraps qihoo360/RzenEmbed to match Qwen3VLEmbedder's interface.

RzenEmbed ships a custom `rzen_embed_inference.py` on its HuggingFace repo and
exposes a single batch API `get_fused_embeddings(instruction, texts, images)`.
This adapter bridges it to the `MediEBEmbeddingModel.encode_input()` contract
which feeds a `List[Dict]` of heterogeneous query/candidate inputs.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Represent the user's input."


class _NoTruncateProcessor:
    """Thin proxy around an HF processor that forces `truncation=False`.

    RzenEmbed's `embed()` hard-codes `truncation=True, max_length=self.max_length`
    when calling its processors. That clips `<|image_pad|>` tokens mid-video
    for long 3D volumes, causing a token-count mismatch between the text
    placeholders and the actual vision features. The error message itself
    suggests "Please disable truncation or increase max_length" — we pick the
    robust option and drop truncation so the full video always fits.

    All other attribute access (e.g. `.tokenizer`, `.image_processor`) passes
    through unchanged via `__getattr__`.
    """

    def __init__(self, processor):
        self._processor = processor

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = False
        kwargs.pop("max_length", None)
        return self._processor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._processor, name)


def _load_rzen_embed(model_name_or_path: str, **hf_kwargs):
    """Instantiate RzenEmbed from the dynamic module shipped with its HF repo.

    RzenEmbed is a plain `nn.Module` (not a `PreTrainedModel`), so we cannot
    use `AutoModel.from_pretrained(trust_remote_code=True)`. Instead we
    snapshot-download the repo and import the class via the same helper HF
    uses internally for trust_remote_code loading.
    """
    local_dir = model_name_or_path
    if not os.path.isdir(local_dir):
        local_dir = snapshot_download(repo_id=model_name_or_path)

    RzenEmbed = get_class_from_dynamic_module(
        "rzen_embed_inference.RzenEmbed", local_dir
    )
    return RzenEmbed(local_dir, **hf_kwargs)


class RzenEmbedEmbedder:
    """Mirror of Qwen3VLEmbedder for the RzenEmbed backbone.

    Exposes `.model`, `.processor`, `.process(list_of_dicts, normalize=True)`
    so `MediEBEmbeddingModel` can use it without other changes.
    """

    def __init__(
        self,
        model_name_or_path: str,
        default_instruction: str = DEFAULT_INSTRUCTION,
        max_frames: int = 64,
        **kwargs,
    ):
        self.default_instruction = default_instruction
        self.max_frames = max_frames

        # RzenEmbed hardcodes torch_dtype=bfloat16 and does not take
        # `torch_dtype` or `device_map` kwargs; strip to avoid TypeError
        # from any future upstream hardening.
        for noisy in ("torch_dtype", "device_map", "low_cpu_mem_usage"):
            kwargs.pop(noisy, None)

        # kwargs still carries attn_implementation + max_length now.
        self.rzen = _load_rzen_embed(model_name_or_path, **kwargs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # RzenEmbed.__init__ already places its submodules on `device`, but
        # calling `.to(device)` again is safe and makes intent explicit.
        self.rzen.to(device)
        self.rzen.eval()
        base = self.rzen.base
        inner = base.model
        if not hasattr(inner, "embed_tokens"):
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "embed_tokens"):
                inner.embed_tokens = inner.language_model.embed_tokens
            else:
                # Fallback: the canonical accessor always works.
                inner.embed_tokens = base.get_input_embeddings()
        if not hasattr(base, "visual") and hasattr(inner, "visual"):
            base.visual = inner.visual

        self.rzen.processor = _NoTruncateProcessor(self.rzen.processor)
        self.rzen.qwen2vl_video_processor = _NoTruncateProcessor(
            self.rzen.qwen2vl_video_processor
        )

        self.model = self.rzen.base
        self.processor = self.rzen.processor

    # ------------------------------------------------------------------
    # Not used at eval (the collator never produces `input_ids`). Raise
    # early to flag misuse during training-path refactors.
    # ------------------------------------------------------------------
    def forward(self, inputs):  # pragma: no cover
        raise NotImplementedError(
            "RzenEmbedEmbedder does not expose a raw tensor forward path. "
            "Eval must go through .process(list_of_input_dicts)."
        )

    # ------------------------------------------------------------------
    # Main batch encoder invoked by MediEBEmbeddingModel.encode_input().
    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Encode a batch of heterogeneous inputs into a [B, D] tensor.

        Strategy: `get_fused_embeddings` accepts a *single* instruction string
        for the whole call and expects a homogeneous modality (all text OR all
        images OR all video-frame lists). We therefore
          1. Bucket samples by modality.
          2. Within each bucket, sub-group by identical `instruction` (in
             practice a batch shares the same instruction per task).
          3. Run each sub-group through `get_fused_embeddings`.
          4. Scatter results back into the original input order.
        """
        out: List[Optional[torch.Tensor]] = [None] * len(inputs)

        # (idx, payload, instruction)
        text_bucket: List[tuple] = []
        image_bucket: List[tuple] = []
        video_bucket: List[tuple] = []

        for i, ele in enumerate(inputs):
            text = ele.get("text")
            image = ele.get("image")
            video = ele.get("video")
            instruction = (ele.get("instruction") or self.default_instruction).strip()

            if video:
                frames = [
                    Image.open(f).convert("RGB") if isinstance(f, str) else f
                    for f in video
                ]
                if self.max_frames and len(frames) > self.max_frames:
                    pick = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
                    frames = [frames[k] for k in pick.tolist()]
                video_bucket.append((i, frames, instruction))
            elif image:
                pil = Image.open(image).convert("RGB") if isinstance(image, str) else image
                image_bucket.append((i, pil, instruction))
            else:
                # text-only (or empty). Use "NULL" sentinel matching
                # Qwen3VLEmbedder's fallback when vision info fails.
                text_bucket.append((i, text if text else "NULL", instruction))

        def _run(bucket, kind):
            if not bucket:
                return
            groups: Dict[str, List[tuple]] = defaultdict(list)
            for idx, payload, ins in bucket:
                groups[ins].append((idx, payload))
            for ins, pairs in groups.items():
                sub_idx = [p[0] for p in pairs]
                sub_items = [p[1] for p in pairs]
                if kind == "text":
                    emb = self.rzen.get_fused_embeddings(
                        instruction=ins, texts=sub_items
                    )
                else:
                    # Images OR video frame-lists go through the `images` arg.
                    # RzenEmbed's internal branching sees List[List[Image]] as video.
                    emb = self.rzen.get_fused_embeddings(
                        instruction=ins, images=sub_items
                    )
                for j, orig_i in enumerate(sub_idx):
                    out[orig_i] = emb[j]

        _run(text_bucket, "text")
        _run(image_bucket, "image")
        _run(video_bucket, "video")

        if any(o is None for o in out):
            missing = [i for i, o in enumerate(out) if o is None]
            raise RuntimeError(f"RzenEmbedEmbedder.process missed samples {missing}")

        embeddings = torch.stack(out, dim=0)
        model_device = self.rzen.base.device
        if embeddings.device != model_device:
            embeddings = embeddings.to(model_device)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
