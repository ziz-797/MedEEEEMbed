"""Adapter that wraps VLM2Vec/VLM2Vec-V2.0 to match Qwen3VLEmbedder's interface.

VLM2Vec-V2.0 is a DoRA/LoRA adapter trained on top of Qwen2-VL-2B-Instruct.
The HF repo ships only an adapter (no `auto_map`, no custom modeling code),
so loading requires `transformers` + `peft` against the base Qwen2-VL weights.

**Checkpoint-vs-library mismatch:** the adapter was saved against an older
Qwen2-VL layout (`base_model.model.model.layers.*`) with single-adapter key
naming (`.lora_A.weight`). `transformers>=4.55` nests the LM under
`model.language_model.*` and `peft>=0.10` appends `.default`, so the official
`PeftModel.from_pretrained` path silently drops all 420 DoRA keys.

We can't downgrade transformers because the Qwen3-VL backbone used elsewhere
in this repo requires transformers>=4.57. So we stay on 4.57 and remap keys
before loading. `_remap_vlm2vec_adapter_keys` is a 1:1 rename; we build the
LoRA structure via `LoraConfig + get_peft_model` and load the remapped
state dict with a strict check, so a future upstream change will surface as
a hard error instead of a silent zero-weight adapter.

Prompt convention (from the repo's inference example):
    - Query (image)   : '<|vision_start|><|image_pad|><|vision_end|> {instruction} {text}'
    - Query (video)   : '<|vision_start|><|video_pad|><|vision_end|> {instruction} {text}'
    - Text candidate  : '{text}' (no instruction prefix)

Embeddings are the last-token hidden state, L2-normalized.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Represent the user's input."
DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_TOKENS = "<|vision_start|><|video_pad|><|vision_end|>"
# Qwen2-VL-2B has max_position_embeddings=32768. At default resolution a
# 64-frame CT volume expands to ~39k vision tokens and triggers a RoPE index
# error. Cap frames and per-frame pixels to match VLM2Vec's official recipe
# (max_pixels=360*420, fps=1). 64 frames * ~150 tokens = ~10k tokens, safe.
MAX_FRAMES = 64
VIDEO_MAX_PIXELS = 360 * 420


def _resolve_base_model(adapter_path: str) -> str:
    """Read `adapter_config.json` to find the base model for this LoRA."""
    local_cfg: Optional[str] = None
    if os.path.isdir(adapter_path):
        candidate = os.path.join(adapter_path, "adapter_config.json")
        if os.path.isfile(candidate):
            local_cfg = candidate
    if local_cfg is None:
        try:
            local_cfg = hf_hub_download(adapter_path, "adapter_config.json")
        except Exception as exc:  # pragma: no cover
            logger.warning("Falling back to default base model (%s): %s", DEFAULT_BASE_MODEL, exc)
            return DEFAULT_BASE_MODEL
    with open(local_cfg, "r") as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)


def _remap_vlm2vec_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rewrite a VLM2Vec-V2 adapter state dict to match modern peft+transformers.

    Mapping (1:1, 420 -> 420):
      base_model.model.model.layers.X.*   -> base_model.model.model.language_model.layers.X.*
      ...lora_A.weight                    -> ...lora_A.default.weight
      ...lora_B.weight                    -> ...lora_B.default.weight
      ...lora_magnitude_vector            -> ...lora_magnitude_vector.default.weight
    """
    remapped: Dict[str, torch.Tensor] = {}
    path_from = "base_model.model.model.layers."
    path_to = "base_model.model.model.language_model.layers."

    for key, value in state_dict.items():
        new_key = key.replace(path_from, path_to, 1) if key.startswith(path_from) else key
        if new_key.endswith(".lora_A.weight") or new_key.endswith(".lora_B.weight"):
            new_key = new_key[: -len(".weight")] + ".default.weight"
        elif new_key.endswith(".lora_magnitude_vector"):
            new_key = new_key + ".default.weight"
        remapped[new_key] = value
    return remapped


def _load_frames(video, max_frames: int, max_pixels: int = VIDEO_MAX_PIXELS) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for f in video:
        img = Image.open(f).convert("RGB") if isinstance(f, str) else f
        frames.append(img)
    if max_frames and len(frames) > max_frames:
        pick = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[k] for k in pick.tolist()]
    if max_pixels:
        capped: List[Image.Image] = []
        for img in frames:
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            capped.append(img)
        frames = capped

    # Qwen2-VL's video processor does `np.stack(frames)`, so every frame in a
    # single volume must have identical (H, W). Real-world 3D datasets (M3D,
    # CT_RATE, etc.) mix scanners so slice sizes vary within one sample. Force
    # every frame onto the first frame's size, matching Qwen3VLEmbedder.
    if frames:
        ref_size = frames[0].size
        frames = [
            f if f.size == ref_size else f.resize(ref_size, Image.BILINEAR)
            for f in frames
        ]
    return frames


class Vlm2VecV2Embedder:
    """Mirror of Qwen3VLEmbedder for the VLM2Vec-V2.0 backbone.

    Exposes `.model`, `.processor`, `.process(list_of_dicts, normalize=True)`
    so `MediEBEmbeddingModel` can use it without other changes.
    """

    def __init__(
        self,
        model_name_or_path: str,
        default_instruction: str = DEFAULT_INSTRUCTION,
        max_frames: int = MAX_FRAMES,
        base_model: Optional[str] = None,
        **kwargs,
    ):
        from peft import LoraConfig, get_peft_model

        self.default_instruction = default_instruction
        self.max_frames = max_frames

        base = base_model or _resolve_base_model(model_name_or_path)
        logger.info("Loading VLM2Vec base=%s adapter=%s", base, model_name_or_path)

        base_model_obj = Qwen2VLForConditionalGeneration.from_pretrained(base, **kwargs)

        adapter_dir = model_name_or_path if os.path.isdir(model_name_or_path) else None
        if adapter_dir is None:
            cfg_path = hf_hub_download(model_name_or_path, "adapter_config.json")
            bin_path = hf_hub_download(model_name_or_path, "adapter_model.bin")
            adapter_dir = os.path.dirname(cfg_path)
        else:
            bin_path = os.path.join(adapter_dir, "adapter_model.bin")

        # Build LoRA structure from config and load remapped DoRA weights.
        # Skips peft's own from_pretrained path so we don't get the misleading
        # 420-key "missing adapter keys" warning.
        lora_cfg = LoraConfig.from_pretrained(adapter_dir)
        self.peft_model = get_peft_model(base_model_obj, lora_cfg)

        raw_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        fixed_sd = _remap_vlm2vec_adapter_keys(raw_sd)
        missing, unexpected = self.peft_model.load_state_dict(fixed_sd, strict=False)
        lora_missing = [k for k in missing if "lora" in k or "magnitude" in k]
        if lora_missing:
            raise RuntimeError(
                f"VLM2Vec adapter remapping still left {len(lora_missing)} "
                f"LoRA keys unloaded (e.g. {lora_missing[:3]})"
            )
        if unexpected:
            raise RuntimeError(
                f"VLM2Vec adapter remapping produced {len(unexpected)} unexpected "
                f"keys (e.g. {unexpected[:3]})"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peft_model.to(device)
        self.peft_model.eval()

        # Processor lives on the adapter repo (has tokenizer + preprocessor).
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, padding_side="left",
        )

        # Expose `.model` for MediEBEmbeddingModel (it reads .device / .config).
        self.model = self.peft_model

    def forward(self, inputs):  # pragma: no cover
        raise NotImplementedError(
            "Vlm2VecV2Embedder does not expose a raw tensor forward path. "
            "Eval must go through .process(list_of_input_dicts)."
        )

    @staticmethod
    def _pool_last(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        flipped = attn_mask.flip(dims=[1])
        last_one = flipped.argmax(dim=1)
        col = attn_mask.shape[1] - last_one - 1
        row = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[row, col]

    def _encode_batch(self, text_list, images=None, videos=None) -> torch.Tensor:
        kw: Dict[str, Any] = {"text": text_list, "return_tensors": "pt", "padding": True}
        if images is not None:
            kw["images"] = images
        if videos is not None:
            kw["videos"] = videos
        inputs = self.processor(**kw)
        inputs = {k: v.to(self.peft_model.device) for k, v in inputs.items()}

        # Skip the lm_head projection — for a 3D volume the sequence length
        # can be tens of thousands of tokens, and `[B, T, vocab=151936]`
        # allocations will OOM even on an 80GB GPU. The underlying
        # `Qwen2VLModel` returns `last_hidden_state` without logits, and
        # peft's LoRA layers still apply because they're wrapped at the
        # Linear level inside the backbone.
        backbone = self.peft_model.get_base_model().model
        with torch.no_grad():
            outputs = backbone(**inputs, return_dict=True)
        hidden = outputs.last_hidden_state
        return self._pool_last(hidden, inputs["attention_mask"])

    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Encode a batch of heterogeneous inputs into a [B, D] tensor.

        Bucket by modality and (within a bucket) by shared instruction so each
        forward pass is homogeneous.
        """
        out: List[Optional[torch.Tensor]] = [None] * len(inputs)

        text_bucket: List[tuple] = []
        image_bucket: List[tuple] = []
        video_bucket: List[tuple] = []

        def _to_pil_list(val) -> List[Image.Image]:
            # Parsers may pass a bare path string, a list of paths (e.g. the
            # image_i2i candidate side always wraps as `[path]`), a PIL image,
            # or a list of PIL images. Normalize everything to List[PIL].
            items = val if isinstance(val, list) else [val]
            return [
                Image.open(x).convert("RGB") if isinstance(x, str) else x
                for x in items
            ]

        for i, ele in enumerate(inputs):
            text = ele.get("text") or ""
            image = ele.get("image")
            video = ele.get("video")
            instruction = (ele.get("instruction") or "").strip()

            if video:
                frames = _load_frames(video, self.max_frames)
                video_bucket.append((i, frames, text, instruction))
            elif image:
                pils = _to_pil_list(image)
                image_bucket.append((i, pils, text, instruction))
            else:
                text_bucket.append((i, text if text else "NULL", instruction))

        def _compose(prefix_tokens: str, instruction: str, text: str) -> str:
            parts = [prefix_tokens] if prefix_tokens else []
            if instruction:
                parts.append(instruction)
            if text:
                parts.append(text)
            return " ".join(parts).strip()

        if text_bucket:
            groups: Dict[str, List[tuple]] = defaultdict(list)
            for idx, t, ins in text_bucket:
                groups[ins].append((idx, t))
            for ins, pairs in groups.items():
                sub_idx = [p[0] for p in pairs]
                sub_text = [_compose("", ins, t) for _, t in pairs]
                emb = self._encode_batch(sub_text)
                for j, orig_i in enumerate(sub_idx):
                    out[orig_i] = emb[j]

        if image_bucket:
            sub_idx = [p[0] for p in image_bucket]
            sub_images = [p[1] for p in image_bucket]  # List[List[PIL]]
            # Multi-image candidates (e.g. image_i2i multi-image branch) need
            # one `<|image_pad|>` per image so the processor's token expansion
            # matches the number of vision features.
            sub_text = [
                _compose(IMAGE_TOKENS * len(p[1]), p[3], p[2]) for p in image_bucket
            ]
            emb = self._encode_batch(sub_text, images=sub_images)
            for j, orig_i in enumerate(sub_idx):
                out[orig_i] = emb[j]

        if video_bucket:
            sub_idx = [p[0] for p in video_bucket]
            sub_videos = [p[1] for p in video_bucket]
            sub_text = [_compose(VIDEO_TOKENS, p[3], p[2]) for p in video_bucket]
            emb = self._encode_batch(sub_text, videos=sub_videos)
            for j, orig_i in enumerate(sub_idx):
                out[orig_i] = emb[j]

        if any(o is None for o in out):
            missing = [i for i, o in enumerate(out) if o is None]
            raise RuntimeError(f"Vlm2VecV2Embedder.process missed samples {missing}")

        embeddings = torch.stack(out, dim=0)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
