"""Image to text: image(+text) query -> text candidates.
Covers classification (image_cls) and VQA (image_qa)."""
import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook


def _resolve_images(image_root, raw_path):
    """Resolve single string or list of paths to full path(s)."""
    if isinstance(raw_path, str):
        return os.path.join(image_root, raw_path)
    return [os.path.join(image_root, p) for p in raw_path]


@add_metainfo_hook
def _prepare(batch_dict, *args, **kwargs):
    image_root = kwargs["image_root"]
    query_inputs, cand_inputs, dataset_infos = [], [], []

    for qry_inst, qry_text, qry_img_path, tgt_texts in zip(
        batch_dict["qry_inst"],
        batch_dict["qry_text"],
        batch_dict["qry_img_path"],
        batch_dict["tgt_text"],
    ):
        query_inputs.append({
            "text": qry_text,
            "image": _resolve_images(image_root, qry_img_path),
            "instruction": qry_inst,
        })
        cand_inputs.append([{"text": t} for t in tgt_texts])
        dataset_infos.append({"cand_names": tgt_texts, "label_name": tgt_texts[0]})

    return {"query_input": query_inputs, "cand_input": cand_inputs, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register("image_cls")
@AutoEvalPairDataset.register("image_qa")
def load_image_to_text(model_args, data_args, *args, **kwargs):
    dataset = load_dataset("json", data_files=kwargs["data_path"], split="train")

    num_sample = kwargs.get("num_sample_per_subset", sys.maxsize)
    if isinstance(num_sample, str) and num_sample.isdigit():
        num_sample = int(num_sample)
    if num_sample < len(dataset):
        dataset = dataset.select(range(num_sample))

    dataset = dataset.map(
        lambda x: _prepare(x, **kwargs),
        batched=True, batch_size=256, num_proc=1,
        drop_last_batch=False, load_from_cache_file=False, keep_in_memory=True,
    )
    return dataset.select_columns(["query_input", "cand_input", "dataset_infos"]), None
