"""Adapter that wraps SigLIP-family dual-tower models.

Tested with:
    - google/medsiglip-448                 (medical-domain fine-tune, 448x448)
    - google/siglip-so400m-patch14-384     (general SigLIP So400m, 384x384)
Any `SiglipModel` checkpoint with the standard text+vision tower pair
should work here.

Unlike Qwen3-VL / RzenEmbed / VLM2Vec (unified decoder-only VLMs),
SigLIP is a **dual-tower** model:
    - `vision_tower`  : single image -> 1152-d embedding
    - `text_tower`    : text (<=64 tokens) -> 1152-d embedding

Both towers land in the same paired-sigmoid contrastive space, so parts
from different modalities can be averaged into one 1152-d vector.

Per sample, every non-empty piece is encoded independently through its
own tower; the final embedding is the mean of those pieces, then
L2-normalized. Pieces are:
    - image         -> vision_tower(image)
    - video         -> mean(per-frame vision_tower(frame))
    - text          -> text_tower(text)          [<=64 tokens]
    - instruction   -> text_tower(instruction)   [<=64 tokens]

Encoding `instruction` as its own text forward (rather than concatenating
with `text`) preserves the 64-token budget for each piece and lets the
task-framing string contribute its own signal.
The same rule is applied to query and candidate sides, so cosine stays
comparable across all pairs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Represent the user's input."
MAX_FRAMES = 64
TEXT_MAX_LENGTH = 64  # SigLIP text tower hard limit


def _to_pil_list(val) -> List[Image.Image]:
    """Normalize parser outputs (str / [str] / PIL / [PIL]) to List[PIL]."""
    items = val if isinstance(val, list) else [val]
    return [Image.open(x).convert("RGB") if isinstance(x, str) else x for x in items]


def _sample_frames(frames: List[Image.Image], max_frames: int) -> List[Image.Image]:
    if not max_frames or len(frames) <= max_frames:
        return frames
    pick = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[k] for k in pick.tolist()]


class SigLipEmbedder:
    """Mirror of the Qwen3VLEmbedder contract for any SigLIP checkpoint.

    Exposes `.model`, `.processor`, `.process(list_of_dicts, normalize=True)`
    so `MediEBEmbeddingModel` uses it without changes.
    """

    def __init__(
        self,
        model_name_or_path: str,
        default_instruction: str = DEFAULT_INSTRUCTION,
        max_frames: int = MAX_FRAMES,
        **kwargs,
    ):
        self.default_instruction = default_instruction
        self.max_frames = max_frames

        # SigLIP ignores attn_implementation for the text tower; strip to
        # avoid unknown-kwarg errors on older transformers builds.
        kwargs.pop("attn_implementation", None)

        self.model = SiglipModel.from_pretrained(model_name_or_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

    def forward(self, inputs):  # pragma: no cover
        raise NotImplementedError(
            "SigLipEmbedder does not expose a raw tensor forward path. "
            "Eval must go through .process(list_of_input_dicts)."
        )

    # ------------------------------------------------------------------
    # Tower encoders — each returns a [B, 1152] tensor.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            return self.model.get_image_features(pixel_values=pixel_values)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LENGTH,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            return self.model.get_text_features(**inputs)

    @torch.no_grad()
    def _encode_videos_meanpool(self, videos: List[List[Image.Image]]) -> torch.Tensor:
        """Per-sample: forward every frame through vision_tower, then mean-pool."""
        flat_frames: List[Image.Image] = []
        offsets: List[int] = [0]
        for frames in videos:
            flat_frames.extend(frames)
            offsets.append(len(flat_frames))
        frame_embeds = self._encode_images(flat_frames)
        pooled = torch.stack(
            [frame_embeds[offsets[i]:offsets[i + 1]].mean(dim=0) for i in range(len(videos))],
            dim=0,
        )
        return pooled

    # ------------------------------------------------------------------
    # Main batch encoder invoked by MediEBEmbeddingModel.encode_input().
    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Encode a batch of heterogeneous inputs into a [B, 1152] tensor.

        Per sample, every non-empty piece (image/video, text, instruction)
        is encoded independently. The sample embedding is the mean of its
        pieces, then L2-normalized.

        Forwards are batched per tower:
          - one vision_tower call for all single-image samples
          - one vision_tower+mean call for all video/multi-image samples
          - one text_tower call batching every text and instruction string
            across all samples
        """
        n = len(inputs)
        # parts[i] accumulates embeddings for sample i
        parts: List[List[torch.Tensor]] = [[] for _ in range(n)]

        image_jobs: List[tuple] = []   # (idx, PIL)
        video_jobs: List[tuple] = []   # (idx, List[PIL])
        text_jobs: List[tuple] = []    # (idx, str)  — covers both text & instruction

        for i, ele in enumerate(inputs):
            text = (ele.get("text") or "").strip()
            instruction = (ele.get("instruction") or "").strip()
            image = ele.get("image")
            video = ele.get("video")

            if video:
                frames = _sample_frames(_to_pil_list(video), self.max_frames)
                video_jobs.append((i, frames))
            elif image:
                pils = _to_pil_list(image)
                # Multi-image samples are treated like videos: per-image
                # vision embedding then mean-pool into one vector.
                if len(pils) == 1:
                    image_jobs.append((i, pils[0]))
                else:
                    video_jobs.append((i, pils))

            if text:
                text_jobs.append((i, text))
            if instruction:
                text_jobs.append((i, instruction))

        # Vision tower (single-image).
        if image_jobs:
            embeds = self._encode_images([p[1] for p in image_jobs])
            for j, (orig_i, _) in enumerate(image_jobs):
                parts[orig_i].append(embeds[j])

        # Vision tower (video / multi-image, per-frame mean-pool).
        if video_jobs:
            embeds = self._encode_videos_meanpool([p[1] for p in video_jobs])
            for j, (orig_i, _) in enumerate(video_jobs):
                parts[orig_i].append(embeds[j])

        # Text tower (batches all text and instruction strings together).
        if text_jobs:
            embeds = self._encode_texts([p[1] for p in text_jobs])
            for j, (orig_i, _) in enumerate(text_jobs):
                parts[orig_i].append(embeds[j])

        # Mean-pool per sample.
        out: List[torch.Tensor] = []
        for i in range(n):
            if not parts[i]:
                # Fully empty sample. Fall back to encoding "NULL" so the
                # batch layout stays intact. Should not normally fire.
                parts[i].append(self._encode_texts(["NULL"])[0])
            out.append(torch.stack(parts[i], dim=0).mean(dim=0))

        embeddings = torch.stack(out, dim=0)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
