# scripts/run_multitarget.py
from __future__ import annotations

import argparse
import json
import os
from typing import List

from biocircuitai.control.multitarget import run_multi_target, TargetSpec

def parse_args():
    p = argparse.ArgumentParser(description="Run multi-target BO + ODE validation.")
    p.add_argument("--model", type=str, default="data/models/surrogate.pt")
    p.add_argument("--outdir", type=str, default="data/multitarget/")
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--hidden", type=str, default="", help="e.g., '256,256' or blank to infer")
    # ways to specify targets
    p.add_argument("--targets", type=str, default="", help="Comma list like '10:8, 1:8, 60:20'")
    p.add_argument("--targets_json", type=str, default="", help="Path to JSON file with [{'A':..,'B':..,'label':'...'}, ...]")
    return p.parse_args()

def _parse_targets_str(s: str) -> List[TargetSpec]:
    t: List[TargetSpec] = []
    if not s.strip():
        return t
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        # formats: "A:B" or "A:B:label"
        parts = [p.strip() for p in pair.split(":")]
        if len(parts) == 2:
            A, B = float(parts[0]), float(parts[1])
            t.append(TargetSpec(A, B, f"A{A}_B{B}"))
        elif len(parts) == 3:
            A, B, label = float(parts[0]), float(parts[1]), parts[2]
            t.append(TargetSpec(A, B, label))
        else:
            raise ValueError(f"Bad target spec '{pair}'. Use 'A:B' or 'A:B:label'.")
    return t

def _parse_targets_json(path: str) -> List[TargetSpec]:
    with open(path, "r") as f:
        raw = json.load(f)
    out: List[TargetSpec] = []
    for item in raw:
        out.append(TargetSpec(float(item["A"]), float(item["B"]), item.get("label")))
    return out

def main():
    args = parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip()) if args.hidden else None

    targets: List[TargetSpec] = []
    if args.targets:
        targets.extend(_parse_targets_str(args.targets))
    if args.targets_json:
        targets.extend(_parse_targets_json(args.targets_json))

    if not targets:
        # sensible defaults demo set
        targets = [
            TargetSpec(10.0, 8.0, "A10_B8"),
            TargetSpec(1.0, 8.0, "A1_B8"),
            TargetSpec(60.0, 20.0, "A60_B20"),
            TargetSpec(5.0, 5.0, "A5_B5"),
        ]

    os.makedirs(args.outdir, exist_ok=True)
    res = run_multi_target(
        model_path=args.model,
        outdir=args.outdir,
        targets=targets,
        trials=args.trials,
        hidden=hidden,
    )
    print(f"[OK] Completed {len(res)} targets. Summary → {os.path.join(args.outdir, 'summary.csv')}")

if __name__ == "__main__":
    main()
