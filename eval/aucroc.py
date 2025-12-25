from pathlib import Path
import argparse
import csv
import json
import os
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Any

try:
    from hamster_tool.tools import open_jsonl  # 可选
except Exception:
    open_jsonl = None

try:
    from verl_hamster.verl.utils.hamster_utils.ece import calculate_aucroc as _calc_aucroc_lib
except Exception:
    _calc_aucroc_lib = None



def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if open_jsonl is not None:
        return open_jsonl(str(path))
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out



def auc_roc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        if j > i:
            mean_rank = (i + j + 2) / 2.0  # 1-based
            ranks[order[i:j + 1]] = mean_rank
        i = j + 1

    sum_ranks_pos = ranks[labels == 1].sum()
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)


def calc_aucroc(conf: List[float], lab: List[int]) -> float:
    scores = np.array([float(x) for x in conf if x is not None], dtype=float)
    labels = np.array([int(b) for b in lab if b is not None], dtype=int)
    if scores.size != labels.size or scores.size == 0:
        return float("nan")

    mask = ~np.isnan(scores)
    scores = scores[mask]
    labels = labels[mask]
    if scores.size == 0:
        return float("nan")

    lo, hi = np.nanmin(scores), np.nanmax(scores)
    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
        scores = (scores - lo) / (hi - lo)

    if _calc_aucroc_lib is not None:
        try:
            return float(_calc_aucroc_lib(scores.tolist(), labels.tolist()))
        except Exception:
            pass
    return auc_roc_numpy(scores, labels)



FILE_PAT = re.compile(r"^(?P<model>.+?)__ece_(?P<dataset>.+?)\.jsonl$", re.IGNORECASE)

def parse_model_dataset(p: Path) -> Tuple[str, str]:
    m = FILE_PAT.match(p.name)
    if m:
        return m.group("model"), m.group("dataset")
    name = p.stem
    if "__ece_" in name:
        model, dataset = name.split("__ece_", 1)
        return model, dataset
    return "unknown_model", name


def collect_conf_and_labels(record: Dict[str, Any], metric: str, direction: str):

    label_raw = record.get("answer_state", None)
    if label_raw is None:
        return None, None
    label = 1 if label_raw is True else 0

    key = f"{metric}_{direction}"
    conf = record.get(key, None)
    if isinstance(conf, (list, tuple)):
        conf = conf[0] if conf else None
    try:
        conf = float(conf) if conf is not None else None
    except Exception:
        conf = None
    return conf, label


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser(description="Batch compute AUROC for ALL models & datasets in directory.")
    parser.add_argument("--results_dir", type=str, default="",
                        help="逐样本结果 jsonl 所在目录（排除 summary 文件）")
    parser.add_argument("--metrics", type=str, default="mean,gmean,rms,max,kl_weighted",
                        help="逗号分隔：mean,gmean,rms,max,kl_weighted")
    parser.add_argument("--directions", type=str, default="forward,backward",
                        help="逗号分隔：forward,backward")
    parser.add_argument("--out_csv", type=str, default="aucroc_summary.csv",
                        help="输出 CSV 文件名（写在 results_dir 下）")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    rows = [] 
    files = [p for p in results_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".jsonl" and "summary" not in p.name.lower()]

    if not files:
        print(f"[ERROR] 目录中未找到样本 jsonl：{results_dir}")
        return

    index: Dict[str, Dict[str, Dict[str, Tuple[float, int, int, int]]]] = {}

    for fp in sorted(files):
        model_tag, dataset_tag = parse_model_dataset(fp)
        records = read_jsonl(fp)
        if not records:
            continue

        labels = []
        for r in records:
            y = r.get("answer_state", None)
            if y is None:
                labels.append(None)
            else:
                labels.append(1 if y is True else 0)

        for metric in metrics:
            for direction in directions:
                confs, labs = [], []
                for r in records:
                    c, y = collect_conf_and_labels(r, metric, direction)
                    if c is None or y is None:
                        continue
                    confs.append(c)
                    labs.append(y)

                n = len(labs)
                pos = int(sum(labs))
                neg = n - pos
                auc = calc_aucroc(confs, labs) if n > 1 else float("nan")

                rows.append({
                    "model": model_tag,
                    "dataset": dataset_tag,
                    "metric": metric,
                    "direction": direction,
                    "n": n,
                    "pos": pos,
                    "neg": neg,
                    "aucroc": auc
                })

                index.setdefault(model_tag, {}).setdefault(dataset_tag, {})[f"{metric}_{direction}"] = (auc, n, pos, neg)

    out_csv = results_dir / args.out_csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "metric", "direction", "n", "pos", "neg", "aucroc"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("=" * 80)
    print(f"AUROC summary written to: {out_csv.resolve()}")
    print("=" * 80)
    for model in sorted(index.keys()):
        print(f"\n### MODEL: {model}")
        datasets = index[model]
        header_cols = ["dataset"]
        for metric in metrics:
            for direction in directions:
                header_cols.append(f"{metric}_{direction}")
        print("\t".join(header_cols))
        for ds in sorted(datasets.keys()):
            cols = [ds]
            md = datasets[ds]
            for metric in metrics:
                for direction in directions:
                    k = f"{metric}_{direction}"
                    auc, n, pos, neg = md.get(k, (float("nan"), 0, 0, 0))
                    val = "nan" if (auc != auc) else f"{auc:.5f}"
                    cols.append(val)
            print("\t".join(cols))


if __name__ == "__main__":
    main()
