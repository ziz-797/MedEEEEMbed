"""Text-to-image retrieval: text query -> image candidates."""
import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook


@add_metainfo_hook
def _prepare(batch_dict, *args, **kwargs):
    image_root = kwargs["image_root"]
    query_inputs, cand_inputs, dataset_infos = [], [], []

    for qry_inst, qry_text, tgt_captions, tgt_img_paths in zip(
        batch_dict["qry_inst"],
        batch_dict["qry_text"],
        batch_dict["tgt_text"],
        batch_dict["tgt_img_path"],
    ):
        query_inputs.append({"text": qry_text, "instruction": qry_inst})
        cand_inputs.append([
            {"text": cap, "image": os.path.join(image_root, p)} if cap else {"image": os.path.join(image_root, p)}
            for cap, p in zip(tgt_captions, tgt_img_paths)
        ])
        dataset_infos.append({"cand_names": tgt_img_paths, "label_name": tgt_img_paths[0]})

    return {"query_input": query_inputs, "cand_input": cand_inputs, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register("image_t2i")
def load_image_t2i(model_args, data_args, *args, **kwargs):
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
