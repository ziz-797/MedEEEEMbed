from typing import Dict, Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoConfig

from ...models.blip2_embedding import Blip2Embedder
from ...models.qwen3_vl_embedding import Qwen3VLEmbedder
from ...models.rzen_embed_embedding import RzenEmbedEmbedder
from ...models.siglip_embedding import SigLipEmbedder
from ...models.vlm2vec_v2_embedding import Vlm2VecV2Embedder


def _select_embedder_cls(model_name_or_path: str):
    """Pick the backbone adapter from the model path / HF repo id."""
    p = (model_name_or_path or "").lower()
    if "blip2" in p or "blip-2" in p:
        return Blip2Embedder
    if "siglip" in p:
        return SigLipEmbedder
    if "vlm2vec" in p:
        return Vlm2VecV2Embedder
    if "rzenembed" in p or p.startswith("qihoo360/") or "/qihoo360/" in p:
        return RzenEmbedEmbedder
    return Qwen3VLEmbedder


class MediEBEmbeddingModel(nn.Module):
    """MediEB embedding wrapper; supports Qwen3-VL or RzenEmbed backbones."""

    def __init__(self,
                 encoder,
                 normalize: bool = True,
                 temperature: float = 0.02):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        # DDP setup
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self):
        return self.encoder.model.device

    @property
    def config(self):
        return self.encoder.model.config

    @classmethod
    def load(cls,
             model_name_or_path: str,
             normalize: bool = True,
             temperature: float = 0.02,
             instruction: Optional[str] = None,
             **kwargs) -> "MediEBEmbeddingModel":
        """Load model from pretrained checkpoint."""
        default_instruction = kwargs.pop('default_instruction', instruction)
        EmbedderCls = _select_embedder_cls(model_name_or_path)
        encoder = EmbedderCls(
            model_name_or_path=model_name_or_path,
            default_instruction=default_instruction or "Represent the user's input.",
            **kwargs
        )
        return cls(encoder=encoder, normalize=normalize, temperature=temperature)

    def save(self, output_dir: str):
        self.encoder.model.save_pretrained(output_dir)
        self.encoder.processor.save_pretrained(output_dir)

    def encode_input(self, inputs: Dict) -> Tensor:
        """Encode inputs using the Qwen3VL embedder.
        
        Args:
            inputs: Dict containing 'text', 'image', 'video', 'instruction' etc.
                    Can be a single dict or a list of dicts.
        """
        # 如果是预处理过的 tensor 输入，直接 forward
        if 'input_ids' in inputs:
            outputs = self.encoder.forward(inputs)
            hidden_state = outputs['last_hidden_state']
            attention_mask = outputs['attention_mask']
            pooled = self._pooling_last(hidden_state, attention_mask)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled
        
        # 否则使用 embedder 的 process 方法
        if isinstance(inputs, dict):
            inputs = [inputs]
        return self.encoder.process(inputs, normalize=self.normalize)

    def _pooling_last(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last non-padded token."""
        last_pos = attention_mask.flip(dims=[1]).argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def forward(self,
                qry: Dict[str, Tensor] = None,
                tgt: Dict[str, Tensor] = None) -> Dict:
        """Forward pass for contrastive learning / evaluation."""
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry = self._dist_gather(qry_reps)
            all_tgt = self._dist_gather(tgt_reps)
        else:
            all_qry, all_tgt = qry_reps, tgt_reps

        scores = torch.matmul(all_qry, all_tgt.T) / self.temperature
        target = torch.arange(scores.size(0), device=scores.device)
        target = target * (all_qry.size(0) // all_tgt.size(0))
        loss = self.cross_entropy(scores, target)

        if self.is_ddp:
            loss = loss * self.world_size
        return {"loss": loss, "qry_reps": qry_reps, "tgt_reps": tgt_reps}

    def _dist_gather(self, t: Tensor) -> Tensor:
        """Gather tensors across distributed processes."""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute similarity matrix between query and passage representations."""
        return torch.matmul(q_reps, p_reps.T)


if __name__ == '__main__':
    import os

    model_path = os.environ.get("MODEL_PATH", "<MODEL_PATH>")
    model = MediEBEmbeddingModel.load(
        model_name_or_path=model_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )

    # Each item is a query or candidate. Allowed keys per dict:
    #   text (str), image (path or URL), video (list of frame paths),
    #   instruction (str)
    inputs = {
        'inputs': [
            {
                'text': 'example query text',
                'instruction': 'Represent the user\'s input.',
            },
            {
                'image': os.environ.get("EXAMPLE_IMAGE", "<IMAGE_PATH>"),
                'instruction': 'Represent the given image.',
            },
        ],
    }

    embeddings = model.encode_input(**inputs)

    print(
        f'Embeddings:\n{embeddings[:, :10].tolist()}\n{embeddings[:, -10:].tolist()}\n'
        f'Score:\n{model.compute_similarity(embeddings, embeddings).tolist()}\n'
    )