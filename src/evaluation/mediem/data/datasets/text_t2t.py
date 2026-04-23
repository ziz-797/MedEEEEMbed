"""Text to text: text query -> text candidates."""
import sys
import json as _json
import re

from datasets import load_dataset, Dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook


def _clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[_-]+', ' ', text)
    text = re.sub(r'\(___, __, __\)', '', text)
    text = re.sub(r'---, ---, ---', '', text)
    text = re.sub(r'\(__, __, ___\)', '', text)
    text = re.sub(r'[_-]+', ' ', text)
    text = re.sub(r'[^\w\s.,:;()\-]', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


@add_metainfo_hook
def _prepare(batch_dict, *args, **kwargs):
    query_inputs, cand_inputs, dataset_infos = [], [], []

    for qry_inst, qry_text, tgt_texts in zip(
        batch_dict['qry_inst'],
        batch_dict['qry_text'],
        batch_dict['tgt_text'],
    ):
        query_inputs.append({
            "text": _clean_text(qry_text),
            "instruction": qry_inst,
        })
        cleaned = [_clean_text(t) for t in tgt_texts]
        cand_inputs.append([{"text": t} for t in cleaned])
        dataset_infos.append({"cand_names": cleaned, "label_name": cleaned[0]})

    return {"query_input": query_inputs, "cand_input": cand_inputs, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register("text_qa")
def load_text_t2t(model_args, data_args, *args, **kwargs):
    data_path = kwargs["data_path"]
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
    except Exception:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = Dataset.from_list(_json.load(f))

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
