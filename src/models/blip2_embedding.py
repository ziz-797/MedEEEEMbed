"""Adapter that wraps Salesforce/blip2-itm-vit-g for retrieval-style embedding.

BLIP-2's ITM variant (`Blip2ForImageTextRetrieval`) exposes two heads:
  - `itm_head` (use_image_text_matching_head=True) — binary match classifier,
    requires a forward pass per (image, text) pair. Not usable as a standalone
    embedding because the score cannot be factored into two cacheable vectors.
  - ITC head (use_image_text_matching_head=False) — cosine of projected
    features; THIS is what we use.

ITC shapes (from transformers' `Blip2ForImageTextRetrieval.forward` source):
  - vision side: ViT-g -> Q-Former(query_embeds=query_tokens, x-attn to image)
                 -> [B, 32, 768] -> vision_projection -> normalize -> [B, 32, 256]
  - text side  : embeddings(input_ids) -> Q-Former(no cross-attn)
                 -> last_hidden_state[:, 0] -> text_projection -> normalize -> [B, 256]

The official similarity is `max_q (image_token_q @ text_cls)` — i.e. each image
carries 32 vectors and retrieval picks the best-matching one. That does not fit
our single-vector cosine framework, so we collapse the 32 tokens to one by
mean-pool then re-normalize. This is the canonical "BLIP-2 as a single-vector
embedder" downgrade.

Per sample, every non-empty piece goes through its own tower, then a mean
across pieces gives the final 256-d vector (matches the SigLIP adapter):
  - image       -> vision tower -> mean over 32 Q tokens -> L2-normalize -> 256-d
  - video       -> mean over per-frame vision embeddings
  - text        -> text tower -> CLS projection -> 256-d
  - instruction -> text tower -> CLS projection -> 256-d (separate forward)
Final embedding = normalize(mean(non-empty parts)).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Represent the user's input."
MAX_FRAMES = 32
TEXT_MAX_LENGTH = 64  # BLIP-2 ITM is trained on short captions; keep budget tight


def _to_pil_list(val) -> List[Image.Image]:
    items = val if isinstance(val, list) else [val]
    return [Image.open(x).convert("RGB") if isinstance(x, str) else x for x in items]


def _sample_frames(frames: List[Image.Image], max_frames: int) -> List[Image.Image]:
    if not max_frames or len(frames) <= max_frames:
        return frames
    pick = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[k] for k in pick.tolist()]


class Blip2Embedder:
    """Mirror of the Qwen3VLEmbedder contract for Blip2ForImageTextRetrieval.

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

        # BLIP-2 ITM isn't attn-implementation-flexible; strip to be safe.
        kwargs.pop("attn_implementation", None)

        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name_or_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

    def forward(self, inputs):  # pragma: no cover
        raise NotImplementedError(
            "Blip2Embedder does not expose a raw tensor forward path. "
            "Eval must go through .process(list_of_input_dicts)."
        )

    # ------------------------------------------------------------------
    # Per-tower encoders. All return [B, 256] L2-normalized.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Reproduces the ITC image path from Blip2ForImageTextRetrieval.forward,
        then mean-pools the 32 Q-Former query tokens to one 256-d vector."""
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.model.device, dtype=self.model.dtype)

        vision_outputs = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device,
        )

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_out = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        image_q = qformer_out.last_hidden_state.to(dtype=self.model.vision_projection.weight.dtype)
        feats = F.normalize(self.model.vision_projection(image_q), dim=-1)  # [B, 32, 256]

        # Collapse 32 query tokens -> 1 via mean, then re-normalize.
        pooled = F.normalize(feats.mean(dim=1), dim=-1)                     # [B, 256]
        return pooled

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Reproduces the ITC text path (no cross-attention) and returns
        the projected CLS embedding."""
        # Bypass Blip2Processor.__call__ for text-only: it tries to compute
        # `max_length - num_query_tokens` which is `int - None` on the ITM
        # checkpoint. The underlying tokenizer works fine on its own.
        enc = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_LENGTH,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        query_embeds = self.model.embeddings(input_ids=input_ids)
        text_out = self.model.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = text_out.last_hidden_state.to(dtype=self.model.text_projection.weight.dtype)
        text_feat = F.normalize(self.model.text_projection(hidden[:, 0, :]), dim=-1)  # [B, 256]
        return text_feat

    @torch.no_grad()
    def _encode_videos_meanpool(self, videos: List[List[Image.Image]]) -> torch.Tensor:
        """Per frame: ViT-g + Q-Former + mean-pool 32 tokens + normalize -> 256-d.
        Per video: mean over frames + re-normalize -> 256-d."""
        flat_frames: List[Image.Image] = []
        offsets: List[int] = [0]
        for frames in videos:
            flat_frames.extend(frames)
            offsets.append(len(flat_frames))
        frame_embeds = self._encode_images(flat_frames)  # [N_frames, 256], already normalized
        pooled = torch.stack(
            [frame_embeds[offsets[i]:offsets[i + 1]].mean(dim=0) for i in range(len(videos))],
            dim=0,
        )
        return F.normalize(pooled, dim=-1)

    # ------------------------------------------------------------------
    # Main batch encoder invoked by MediEBEmbeddingModel.encode_input().
    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Per sample, every non-empty piece is encoded independently. The
        sample embedding is the mean of its pieces, L2-normalized."""
        n = len(inputs)
        parts: List[List[torch.Tensor]] = [[] for _ in range(n)]

        image_jobs: List[tuple] = []   # (idx, PIL)
        video_jobs: List[tuple] = []   # (idx, List[PIL])
        text_jobs: List[tuple] = []    # (idx, str)

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
                if len(pils) == 1:
                    image_jobs.append((i, pils[0]))
                else:
                    video_jobs.append((i, pils))

            if text:
                text_jobs.append((i, text))
            if instruction:
                text_jobs.append((i, instruction))

        if image_jobs:
            embeds = self._encode_images([p[1] for p in image_jobs])
            for j, (orig_i, _) in enumerate(image_jobs):
                parts[orig_i].append(embeds[j])

        if video_jobs:
            embeds = self._encode_videos_meanpool([p[1] for p in video_jobs])
            for j, (orig_i, _) in enumerate(video_jobs):
                parts[orig_i].append(embeds[j])

        if text_jobs:
            embeds = self._encode_texts([p[1] for p in text_jobs])
            for j, (orig_i, _) in enumerate(text_jobs):
                parts[orig_i].append(embeds[j])

        out: List[torch.Tensor] = []
        for i in range(n):
            if not parts[i]:
                parts[i].append(self._encode_texts(["NULL"])[0])
            out.append(torch.stack(parts[i], dim=0).mean(dim=0))

        embeddings = torch.stack(out, dim=0)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
