#!/usr/bin/env python3
"""
Medical Evaluation Results Collection Script
Collect and summarize medical multimodal evaluation task results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse


# Define medical task category configurations
TASK_CATEGORIES = {
    'MED_2D_CLS': {
        'metric': 'hit@1',
        'domain': '2D_Task',
        'tasks': [
            'MIMIC-CXR-LT_cls', 'ISIC-2019_cls',
            'Brain-Tumor-MRI_cls', 'BloodMNIST_cls', 'DermaMNIST_cls',
            'OCTMNIST_cls', 'OrganAMNIST_cls', 'OrganCMNIST_cls',
            'OrganSMNIST_cls', 'PathMNIST_cls', 'TissueMNIST_cls',
            'Kvasir_cls', 'APTOS_cls', 'ChestMNIST_cls',
        ]
    },
    'MED_2D_I2I': {
        'metric': 'hit@1',
        'domain': '2D_Task',
        'tasks': [
            'MIMIC-CXR-LT_i2i', 'ISIC-2019_i2i',
            'Brain-Tumor-MRI_i2i', 'BloodMNIST_i2i', 'DermaMNIST_i2i',
            'OCTMNIST_i2i', 'OrganAMNIST_i2i', 'OrganCMNIST_i2i',
            'OrganSMNIST_i2i', 'PathMNIST_i2i', 'TissueMNIST_i2i',
            'Kvasir_i2i', 'PanNuke_i2i', 'APTOS_i2i', 'ChestMNIST_i2i',
        ]
    },
    'MED_2D_T2I': {
        'metric': 'hit@1',
        'domain': '2D_Task',
        'tasks': [
            'MIMIC-CXR-LT_t2i', 'ISIC-2019_t2i',
            'Brain-Tumor-MRI_t2i', 'BloodMNIST_t2i', 'DermaMNIST_t2i',
            'OCTMNIST_t2i', 'OrganAMNIST_t2i', 'OrganCMNIST_t2i',
            'OrganSMNIST_t2i', 'PathMNIST_t2i', 'TissueMNIST_t2i',
            'Kvasir_t2i', 'PanNuke_t2i', 'APTOS_t2i', 'ChestMNIST_t2i',
        ]
    },
    'MED_2D_I2T': {
        'metric': 'hit@1',
        'domain': '2D_Task',
        'tasks': ['MIMIC_CXR_report', 'USData_report'],
    },
    'MED_2D_VQA': {
        'metric': 'hit@1',
        'domain': '2D_Task',
        'tasks': [
            'Path-VQA', 'PMC-VQA', 'ROCO-VQA',
            'MedPIX', 'RadLmageNet', 'VQA_RAD', 'MIMIC-CXR-VQA',
        ]
    },
    'MED_2D_VG': {
        'metric': 'hit@1',
        'domain': '2D_Task',
        'tasks': [
            'PanNuke_VG', 'UltrasoundNerve',
            'ChestImagenome', 'Gastrointestinal',
            'SkinLesion', 'VindrCXR', 'VindrMammo',
        ]
    },
    'MED_T2T': {
        'metric': 'hit@1',
        'domain': 'Text_Task',
        'tasks': [
            'MedMCQA', 'MIMIC_Findings_Impression', 'PubMedQA',
            'MedicalQA', 'PublicHealthQA', 'MMDental',
        ]
    },
    'MED_3D_CLS': {
        'metric': 'hit@1',
        'domain': '3D_Task',
        'tasks': ['CT_RATE_cls', 'ChirrMRI600_cls', 'MRNet_cls',
                  'NoduleMNIST_cls', 'Organ3dMNIST_cls', 'SynapseMNIST_cls']
    },
    'MED_3D_VQA': {
        'metric': 'hit@1',
        'domain': '3D_Task',
        'tasks': ['CT_RATE_vqa', 'RGCC_vqa', 'M3D_3dqa']
    },
    'MED_3D_I2T': {
        'metric': 'hit@1',
        'domain': '3D_Task',
        'tasks': ['RGCC_i2t', 'M3D_i2t', 'MMDental_i2t']
    },
    'MED_3D_I2I': {
        'metric': 'hit@1',
        'domain': '3D_Task',
        'tasks': ['CT_RATE_i2i', 'ChirrMRI600_i2i', 'MRNet_i2i', 'Organ3dMNIST_i2i',
                  'BraTS_t1_to_t2', 'BraTS_t2_to_t1',
                  'ChirrMRI600_t1_to_t2', 'ChirrMRI600_t2_to_t1',
                  'HaNSeg_mri_to_ct',
                  'SynthRAD_brain_ct2mri', 'SynthRAD_brain_mri2ct',
                  'SynthRAD_pelvis_ct2mri', 'SynthRAD_pelvis_mri2ct']
    },
    'MED_3D_T2I': {
        'metric': 'hit@1',
        'domain': '3D_Task',
        'tasks': ['CT_RATE_t2i', 'ChirrMRI600_t2i', 'M3D_t2i',
                  'MRNet_t2i', 'Organ3dMNIST_t2i']
    },
}


SUMMARY_GROUPS = {
    '2D':  ['MED_2D_CLS', 'MED_2D_I2I', 'MED_2D_T2I', 'MED_2D_I2T', 'MED_2D_VQA', 'MED_2D_VG'],
    'TXT': ['MED_T2T'],
    '3D':  ['MED_3D_CLS', 'MED_3D_VQA', 'MED_3D_I2T', 'MED_3D_T2I', 'MED_3D_I2I'],
}


def load_score(eval_dir: Path, domain: str, task: str) -> Optional[Dict]:
    """Load score results for a single task"""
    score_file = eval_dir / domain / f"{task}_score.json"
    if not score_file.exists():
        return None
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Failed to read {score_file}: {e}")
        return None


def collect_results(eval_dir: Path) -> tuple:
    """Collect all evaluation results"""
    category_results = {}
    task_results = {}

    for category, config in TASK_CATEGORIES.items():
        metric = config['metric']
        domain = config['domain']
        tasks = config['tasks']

        scores = []
        for task in tasks:
            score_data = load_score(eval_dir, domain, task)
            if score_data and metric in score_data:
                score = score_data[metric] * 100
                scores.append(score)
                task_results[task] = {metric: score}

        if scores:
            category_results[category] = sum(scores) / len(scores)
        else:
            category_results[category] = None

    return category_results, task_results


def compute_summary(category_results: Dict, task_results: Dict) -> Dict:
    """Compute summary results"""
    summary = {}

    # Copy category results
    for cat, val in category_results.items():
        if val is not None:
            summary[cat] = val

    for group_name, cats in SUMMARY_GROUPS.items():
        all_scores = []
        for cat in cats:
            for task in TASK_CATEGORIES[cat]['tasks']:
                if task in task_results:
                    metric_key = list(task_results[task].keys())[0]
                    all_scores.append(task_results[task][metric_key])
        if all_scores:
            summary[group_name] = sum(all_scores) / len(all_scores)

    # ALL
    all_scores = [v for scores in task_results.values() for v in scores.values()]
    if all_scores:
        summary['ALL'] = sum(all_scores) / len(all_scores)

    return summary


def print_table(headers, rows, title="", max_width=120):
    """Print table in terminal, split into multiple lines if too wide"""
    if title:
        print(f"\n{'=' * max_width}")
        print(f"{title:^{max_width}}")
        print(f"{'=' * max_width}")

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    chunks = []
    current_chunk = []
    current_width = 0

    for i, (header, width) in enumerate(zip(headers, col_widths)):
        needed_width = width + (3 if current_chunk else 0)
        if current_width + needed_width > max_width and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [i]
            current_width = width
        else:
            current_chunk.append(i)
            current_width += needed_width

    if current_chunk:
        chunks.append(current_chunk)

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx > 0:
            print()

        chunk_headers = [headers[i] for i in chunk]
        chunk_widths = [col_widths[i] for i in chunk]

        header_line = " | ".join(h.ljust(w) for h, w in zip(chunk_headers, chunk_widths))
        print(f"\n{header_line}")
        print("-" * len(header_line))

        for row in rows:
            chunk_cells = [row[i] for i in chunk]
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(chunk_cells, chunk_widths))
            print(row_line)

    print()


def save_tsv(file_path: Path, headers, rows):
    """Save as TSV file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(headers) + '\n')
        for row in rows:
            f.write('\t'.join(str(cell) for cell in row) + '\n')
    print(f"Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect medical evaluation results')
    parser.add_argument('eval_dir', type=str, help='Evaluation results directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (optional)')
    parser.add_argument('--max_width', type=int, default=120, help='Maximum terminal width')

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory does not exist {eval_dir}")
        return

    print("Collecting medical evaluation results...")
    category_results, task_results = collect_results(eval_dir)
    summary = compute_summary(category_results, task_results)

    # Summary table
    summary_order = [
        'MED_2D_CLS', 'MED_2D_I2I', 'MED_2D_T2I', 'MED_2D_I2T', 'MED_2D_VQA', 'MED_2D_VG', '2D',
        'MED_T2T', 'TXT',
        'MED_3D_CLS', 'MED_3D_VQA', 'MED_3D_I2T', 'MED_3D_T2I', 'MED_3D_I2I', '3D',
        'ALL',
    ]

    # Count tasks per category
    cat_counts = {}
    for cat in summary_order:
        if cat in TASK_CATEGORIES:
            cat_counts[cat] = len(TASK_CATEGORIES[cat]['tasks'])
        elif cat in SUMMARY_GROUPS:
            cat_counts[cat] = sum(len(TASK_CATEGORIES[c]['tasks']) for c in SUMMARY_GROUPS[cat])
        elif cat == 'ALL':
            cat_counts[cat] = sum(len(c['tasks']) for c in TASK_CATEGORIES.values())

    present = [cat for cat in summary_order if cat in summary]
    summary_headers = [f"{cat}({cat_counts.get(cat, 0)})" for cat in present]
    summary_row = [f"{summary[cat]:.1f}" for cat in present]

    print_table(summary_headers, [summary_row], "Medical Evaluation Summary", max_width=args.max_width)

    # Details table
    all_tasks = []
    for config in TASK_CATEGORIES.values():
        all_tasks.extend(config['tasks'])

    details_headers = []
    details_row = []
    for task in all_tasks:
        if task in task_results:
            metric_key = list(task_results[task].keys())[0]
            score = task_results[task][metric_key]
            details_headers.append(task)
            details_row.append(f"{score:.1f}")

    if details_headers:
        print_table(details_headers, [details_row], "Detailed Results (Hit@1 x100)", max_width=args.max_width)

    # File output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_tsv(output_dir / 'med_summary.tsv', summary_headers, [summary_row])
        if details_headers:
            save_tsv(output_dir / 'med_details.tsv', details_headers, [details_row])

        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
