#!/usr/bin/env python3
"""
Collect medical evaluation scores across multiple models into a CSV.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List

from .gather_med_results import (
    RETRIEVAL_FAMILIES,
    SUMMARY_GROUPS,
    TASK_CATEGORIES,
    load_score,
)


CATEGORY_ORDER = [
    'MED_2D_CLS', 'MED_2D_I2I', 'MED_2D_T2I', 'MED_2D_I2T', 'MED_2D_VQA', 'MED_2D_VG',
    'MED_T2T',
    'MED_3D_CLS', 'MED_3D_VQA', 'MED_3D_I2T', 'MED_3D_T2I', 'MED_3D_I2I',
    'MED_OOD_CXR', 'MED_OOD_Retinal', 'MED_OOD_LC25000',
    'MED_OOD_OmniMedVQA', 'MED_OOD_T2T', 'MED_OOD_BraTS_MEN',
]
SUMMARY_ORDER = ['ALL', '2D', 'TXT', '3D', 'OOD', 'ALL+OOD']
FAMILY_ORDER = [
    'FAM_I2I', 'FAM_CrossMod', 'FAM_CLS', 'FAM_I2T',
    'FAM_T2I', 'FAM_VQA', 'FAM_VG', 'FAM_T2T',
]


def collect_model(eval_dir: Path, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for config in TASK_CATEGORIES.values():
        for task in config['tasks']:
            score = load_score(eval_dir, config['domain'], task)
            if not score:
                continue
            vals = {m: score[m] * 100 for m in metrics if m in score}
            if vals:
                out[task] = vals
    return out


def avg(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def category_score(scores: Dict[str, Dict[str, float]], cat: str, metric: str) -> float | None:
    return avg([scores[t][metric] for t in TASK_CATEGORIES[cat]['tasks']
                if t in scores and metric in scores[t]])


def family_score(scores: Dict[str, Dict[str, float]], fam: str, metric: str) -> float | None:
    return avg([scores[t][metric] for t in RETRIEVAL_FAMILIES[fam]['tasks']
                if t in scores and metric in scores[t]])


def super_score(scores: Dict[str, Dict[str, float]], group: str, metric: str) -> float | None:
    if group == 'ALL':
        # In-distribution only: OOD reported separately
        tasks = [t for g, cats in SUMMARY_GROUPS.items() if g != 'OOD'
                 for cat in cats for t in TASK_CATEGORIES[cat]['tasks']]
    elif group == 'ALL+OOD':
        tasks = [t for cfg in TASK_CATEGORIES.values() for t in cfg['tasks']]
    else:
        tasks = [t for cat in SUMMARY_GROUPS[group] for t in TASK_CATEGORIES[cat]['tasks']]
    return avg([scores[t][metric] for t in tasks if t in scores and metric in scores[t]])


def n_tasks(name: str) -> int:
    if name in TASK_CATEGORIES:
        return len(TASK_CATEGORIES[name]['tasks'])
    if name in RETRIEVAL_FAMILIES:
        return len(RETRIEVAL_FAMILIES[name]['tasks'])
    if name in SUMMARY_GROUPS:
        return sum(len(TASK_CATEGORIES[c]['tasks']) for c in SUMMARY_GROUPS[name])
    if name == 'ALL':
        # In-distribution only
        return sum(
            len(TASK_CATEGORIES[c]['tasks'])
            for g, cats in SUMMARY_GROUPS.items() if g != 'OOD'
            for c in cats
        )
    if name == 'ALL+OOD':
        return sum(len(cfg['tasks']) for cfg in TASK_CATEGORIES.values())
    return 1


def task_to_category(task: str) -> str:
    for cat, cfg in TASK_CATEGORIES.items():
        if task in cfg['tasks']:
            return cat
    return ''


def fmt(v: float | None) -> str:
    return f'{v:.2f}' if v is not None else ''


def build_rows(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    metrics: List[str],
) -> List[List[str]]:
    rows: List[List[str]] = []

    # summary rows
    for g in SUMMARY_ORDER:
        row = ['summary', g, '', str(n_tasks(g))]
        for m in models:
            for metric in metrics:
                row.append(fmt(super_score(all_scores.get(m, {}), g, metric)))
        rows.append(row)

    # family rows (orthogonal partition by retrieval direction)
    for fam in FAMILY_ORDER:
        row = ['family', fam, '', str(n_tasks(fam))]
        for m in models:
            for metric in metrics:
                row.append(fmt(family_score(all_scores.get(m, {}), fam, metric)))
        rows.append(row)

    # category rows
    for cat in CATEGORY_ORDER:
        if cat not in TASK_CATEGORIES:
            continue
        row = ['category', cat, '', str(n_tasks(cat))]
        for m in models:
            for metric in metrics:
                row.append(fmt(category_score(all_scores.get(m, {}), cat, metric)))
        rows.append(row)

    # task rows
    for cat in CATEGORY_ORDER:
        if cat not in TASK_CATEGORIES:
            continue
        for task in TASK_CATEGORIES[cat]['tasks']:
            if not any(task in all_scores[m] for m in models):
                continue
            row = ['task', task, cat, '1']
            for m in models:
                for metric in metrics:
                    row.append(fmt(all_scores.get(m, {}).get(task, {}).get(metric)))
            rows.append(row)

    return rows


def discover_models(results_root: Path) -> List[str]:
    return sorted([d.name for d in results_root.iterdir()
                   if d.is_dir() and any(d.rglob('*_score.json'))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results_root', type=str)
    ap.add_argument('-o', '--output', type=str, default=None,
                    help='CSV output path. Default: <results_root>/comparison.csv')
    ap.add_argument('--models', type=str, default=None,
                    help='Comma-separated model dirs. Auto-discover if absent.')
    ap.add_argument('--metrics', type=str, default='hit@1',
                    help='Comma-separated metric keys, e.g. hit@1,hit@5,hit@10')
    ap.add_argument('--sort-by', type=str, default=None,
                    help='Put this model first in column order.')
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise SystemExit(f'Not found: {root}')

    models = [m.strip() for m in args.models.split(',')] if args.models else discover_models(root)
    if args.sort_by and args.sort_by in models:
        models = [args.sort_by] + [m for m in models if m != args.sort_by]

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    all_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in models:
        ed = root / name
        if not ed.exists():
            print(f'[warn] missing: {ed}')
            continue
        ts = collect_model(ed, metrics)
        if ts:
            all_scores[name] = ts
        else:
            print(f'[warn] no scores under {ed}')

    if not all_scores:
        raise SystemExit('No results collected.')

    models = [m for m in models if m in all_scores]
    headers = ['level', 'name', 'parent_category', 'n_tasks']
    for m in models:
        for metric in metrics:
            headers.append(f'{m}::{metric}' if len(metrics) > 1 else m)

    rows = build_rows(all_scores, models, metrics)

    out_path = Path(args.output) if args.output else root / 'comparison.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f'Wrote {len(rows)} rows ({len(models)} models × {len(metrics)} metric(s)) to {out_path}')


if __name__ == '__main__':
    main()
