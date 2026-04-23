"""Image-to-image retrieval: image query -> image candidates."""
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

    for qry_inst, qry_text, qry_img_path, tgt_captions, tgt_img_paths in zip(
        batch_dict["qry_inst"],
        batch_dict["qry_text"],
        batch_dict["qry_img_path"],
        batch_dict["tgt_text"],
        batch_dict["tgt_img_path"],
    ):
        query_inputs.append({
            "text": qry_text,
            "image": _resolve_images(image_root, qry_img_path),
            "instruction": qry_inst,
        })
        cand_img_paths = [os.path.join(image_root, p) for p in tgt_img_paths]

        if len(tgt_captions) == len(cand_img_paths):
            # 1:1 — each image is a separate candidate
            cand_inputs.append([
                {"text": cap, "image": [img]} for cap, img in zip(tgt_captions, cand_img_paths)
            ])
            cand_names = [f"{p}:{c}" for p, c in zip(tgt_img_paths, tgt_captions)]
        else:
            # All images form one multi-image candidate
            cand_item = {"image": cand_img_paths}
            if tgt_captions and tgt_captions[0]:
                cand_item["text"] = tgt_captions[0]
            cand_inputs.append([cand_item])
            cand_names = ["+".join(tgt_img_paths)]

        dataset_infos.append({"cand_names": cand_names, "label_name": cand_names[0]})

    return {"query_input": query_inputs, "cand_input": cand_inputs, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register("image_i2i")
@AutoEvalPairDataset.register("image_i2i_vg")
def load_image_i2i(model_args, data_args, *args, **kwargs):
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
