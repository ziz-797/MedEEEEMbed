"""Text to 3D volume: text query -> volume (video) candidates.
Each candidate is a list of slice images processed as video."""
import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook


@add_metainfo_hook
def _prepare(batch_dict, *args, **kwargs):
    image_root = kwargs["image_root"]
    query_inputs, cand_inputs, dataset_infos = [], [], []

    for qry_inst, qry_text, tgt_texts, tgt_img_groups in zip(
        batch_dict["qry_inst"],
        batch_dict["qry_text"],
        batch_dict["tgt_text"],
        batch_dict["tgt_img_path"],
    ):
        query_inputs.append({"text": qry_text, "instruction": qry_inst})

        current_cands = []
        current_names = []
        for i, img_group in enumerate(tgt_img_groups):
            # Each candidate is a list of slice paths -> video
            paths = img_group if isinstance(img_group, list) else [img_group]
            full_paths = [os.path.join(image_root, p) for p in paths]
            cand_item = {"video": full_paths}
            if i < len(tgt_texts) and tgt_texts[i]:
                cand_item["text"] = tgt_texts[i]
            current_cands.append(cand_item)
            current_names.append("+".join(paths[:3]) + "...")  # short name for dedup

        cand_inputs.append(current_cands)
        dataset_infos.append({"cand_names": current_names, "label_name": current_names[0]})

    return {"query_input": query_inputs, "cand_input": cand_inputs, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register("volume_t2i")
def load_volume_t2i(model_args, data_args, *args, **kwargs):
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
