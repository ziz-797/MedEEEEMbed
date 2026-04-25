#!/usr/bin/env python3
"""Produce a medical-evaluation leaderboard report (JSON + Markdown).

Reads per-model results from <RESULTS_ROOT> (one sub-directory per model) and
emits:
  1. A JSON leaderboard keyed by model, containing summary / category /
     per-task hit@1 (plus optional extra metrics).
  2. A Markdown table view of the same data, ready to paste into a doc.

Example:
  python -m src.evaluation.mediem.report_med <RESULTS_ROOT> \
      --metrics hit@1,hit@5,hit@10 \
      --models <MODEL_NAME_A>,<MODEL_NAME_B> \
      --out-json report.json \
      --out-md report.md

If --models is omitted, every sub-directory containing `*_score.json` is
picked up automatically.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .gather_med_results import TASK_CATEGORIES, SUMMARY_GROUPS, load_score


CATEGORY_ORDER = [
    'MED_2D_CLS', 'MED_2D_I2I', 'MED_2D_T2I', 'MED_2D_I2T', 'MED_2D_VQA', 'MED_2D_VG',
    'MED_T2T',
    'MED_3D_CLS', 'MED_3D_VQA', 'MED_3D_I2T', 'MED_3D_T2I', 'MED_3D_I2I',
    'MED_OOD_CXR', 'MED_OOD_Retinal', 'MED_OOD_LC25000',
    'MED_OOD_OmniMedVQA', 'MED_OOD_BraTS_MEN',
]
SUMMARY_ORDER = ['ALL', '2D', 'TXT', '3D', 'OOD', 'ALL+OOD']


def collect_model(eval_dir: Path, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cfg in TASK_CATEGORIES.values():
        for task in cfg['tasks']:
            score = load_score(eval_dir, cfg['domain'], task)
            if not score:
                continue
            row = {m: score[m] * 100 for m in metrics if m in score}
            if 'num_data' in score:
                row['n'] = int(score['num_data'])
            if row:
                out[task] = row
    return out


def _avg(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def category_score(task_scores: Dict[str, Dict[str, float]], cat: str, metric: str) -> Optional[float]:
    vals = [task_scores[t][metric] for t in TASK_CATEGORIES[cat]['tasks']
            if t in task_scores and metric in task_scores[t]]
    return _avg(vals)


def super_score(task_scores: Dict[str, Dict[str, float]], group: str, metric: str) -> Optional[float]:
    if group == 'ALL':
        # In-distribution only (OOD summary group excluded)
        tasks = [t for g, cats in SUMMARY_GROUPS.items() if g != 'OOD'
                 for cat in cats for t in TASK_CATEGORIES[cat]['tasks']]
    elif group == 'ALL+OOD':
        tasks = [t for cfg in TASK_CATEGORIES.values() for t in cfg['tasks']]
    else:
        tasks = [t for cat in SUMMARY_GROUPS[group] for t in TASK_CATEGORIES[cat]['tasks']]
    vals = [task_scores[t][metric] for t in tasks
            if t in task_scores and metric in task_scores[t]]
    return _avg(vals)


def task_counts() -> Dict[str, int]:
    counts = {cat: len(cfg['tasks']) for cat, cfg in TASK_CATEGORIES.items()}
    for group, cats in SUMMARY_GROUPS.items():
        counts[group] = sum(counts[c] for c in cats)
    # ALL is in-distribution only; ALL+OOD is the full 100-task total
    counts['ALL'] = sum(
        len(TASK_CATEGORIES[c]['tasks'])
        for g, cats in SUMMARY_GROUPS.items() if g != 'OOD'
        for c in cats
    )
    counts['ALL+OOD'] = sum(len(cfg['tasks']) for cfg in TASK_CATEGORIES.values())
    return counts


def task_to_category(task: str) -> str:
    for cat, cfg in TASK_CATEGORIES.items():
        if task in cfg['tasks']:
            return cat
    return ''


def build_report(
    results_root: Path,
    models: List[str],
    metrics: List[str],
) -> Dict:
    primary_metric = metrics[0]
    counts = task_counts()
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'results_root': str(results_root),
        'primary_metric': primary_metric,
        'metrics': metrics,
        'task_counts': {k: counts[k] for k in SUMMARY_ORDER + CATEGORY_ORDER},
        'models': {},
    }
    for name in models:
        eval_dir = results_root / name
        if not eval_dir.exists():
            print(f'[warn] missing: {eval_dir}')
            continue
        task_scores = collect_model(eval_dir, metrics)
        if not task_scores:
            print(f'[warn] no scores under {eval_dir}')
            continue
        block = {'summary': {}, 'category': {}, 'task': {}}
        for m in metrics:
            block['summary'][m] = {g: super_score(task_scores, g, m) for g in SUMMARY_ORDER}
            block['category'][m] = {
                c: category_score(task_scores, c, m) for c in CATEGORY_ORDER
                if c in TASK_CATEGORIES
            }
        for task, row in sorted(task_scores.items()):
            block['task'][task] = {'parent_category': task_to_category(task), **row}
        report['models'][name] = block
    return report


def format_cell(v: Optional[float]) -> str:
    return '—' if v is None else f'{v:.2f}'


def format_markdown(report: Dict) -> str:
    models = list(report['models'].keys())
    if not models:
        return '# No results.'
    m = report['primary_metric']
    counts = report['task_counts']
    lines = [
        f'# Medical Evaluation Report',
        f'',
        f'- Generated: `{report["generated_at"]}`',
        f'- Results root: `{report["results_root"]}`',
        f'- Primary metric: **{m} × 100**',
        '',
        '## Summary',
        '',
    ]
    header = ['Category (n)'] + models
    sep = ['---'] + ['---:'] * len(models)
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(sep) + ' |')
    for g in SUMMARY_ORDER:
        row = [f'**{g}** ({counts[g]})']
        for name in models:
            v = report['models'][name]['summary'][m].get(g)
            row.append(f'**{format_cell(v)}**' if v is not None else '—')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('## Category breakdown')
    lines.append('')
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(sep) + ' |')
    for c in CATEGORY_ORDER:
        row = [f'{c} ({counts[c]})']
        for name in models:
            v = report['models'][name]['category'][m].get(c)
            row.append(format_cell(v))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('## Per-task detail')
    lines.append('')
    lines.append('| Category | Task | ' + ' | '.join(models) + ' |')
    lines.append('| --- | --- | ' + ' | '.join(['---:'] * len(models)) + ' |')
    all_tasks = [t for cat in CATEGORY_ORDER if cat in TASK_CATEGORIES
                 for t in TASK_CATEGORIES[cat]['tasks']]
    for task in all_tasks:
        cat = task_to_category(task)
        row = [cat, task]
        any_val = False
        for name in models:
            v = report['models'][name]['task'].get(task, {}).get(m)
            row.append(format_cell(v))
            if v is not None:
                any_val = True
        if any_val:
            lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines) + '\n'


def discover_models(root: Path) -> List[str]:
    return sorted([d.name for d in root.iterdir()
                   if d.is_dir() and any(d.rglob('*_score.json'))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results_root')
    ap.add_argument('--models', default=None, help='Comma-separated; auto-discover if absent.')
    ap.add_argument('--metrics', default='hit@1,hit@5,hit@10',
                    help='Comma-separated metric keys. First one drives the leaderboard.')
    ap.add_argument('--sort-by', default=None, help='Put this model first in column order.')
    ap.add_argument('--out-json', default=None, help='Path to write JSON leaderboard.')
    ap.add_argument('--out-md', default=None, help='Path to write Markdown leaderboard.')
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise SystemExit(f'Not found: {root}')

    models = [m.strip() for m in args.models.split(',')] if args.models else discover_models(root)
    if args.sort_by and args.sort_by in models:
        models = [args.sort_by] + [m for m in models if m != args.sort_by]
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    report = build_report(root, models, metrics)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f'wrote JSON  -> {args.out_json}')
    if args.out_md:
        Path(args.out_md).write_text(format_markdown(report))
        print(f'wrote Markdown -> {args.out_md}')
    if not args.out_json and not args.out_md:
        print(format_markdown(report))


if __name__ == '__main__':
    main()
