"""3D volume to text: multi-image (video) query -> text candidates.
Covers 3D classification, VQA, and image-to-text tasks."""
import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook


@add_metainfo_hook
def _prepare(batch_dict, *args, **kwargs):
    image_root = kwargs["image_root"]
    query_inputs, cand_inputs, dataset_infos = [], [], []

    modalities = batch_dict.get("qry_modality", ["video"] * len(batch_dict["qry_inst"]))

    for qry_inst, qry_text, qry_img_paths, modality, tgt_texts in zip(
        batch_dict["qry_inst"],
        batch_dict["qry_text"],
        batch_dict["qry_img_path"],
        modalities,
        batch_dict["tgt_text"],
    ):
        paths = qry_img_paths if isinstance(qry_img_paths, list) else [qry_img_paths]
        full_paths = [os.path.join(image_root, p) for p in paths]

        query_item = {"text": qry_text, "instruction": qry_inst}
        if modality == "image":
            query_item["image"] = full_paths if len(full_paths) > 1 else full_paths[0]
        else:
            query_item["video"] = full_paths
        query_inputs.append(query_item)
        tgt_texts = list(tgt_texts) if isinstance(tgt_texts, (list, tuple)) else [tgt_texts]
        cand_inputs.append([{"text": t} for t in tgt_texts])
        dataset_infos.append({"cand_names": tgt_texts, "label_name": tgt_texts[0]})

    return {"query_input": query_inputs, "cand_input": cand_inputs, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register("volume_i2t")
def load_volume_i2t(model_args, data_args, *args, **kwargs):
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
