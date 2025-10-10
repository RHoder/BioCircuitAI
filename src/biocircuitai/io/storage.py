# src/biocircuitai/io/storage.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json, csv, os
from typing import Dict, Any

REGISTRY_CSV = Path("data/registry.csv")

@dataclass
class RunRecord:
    run_id: str
    phase: str                     # "sweep" | "train" | "bo" | "validate"
    status: str                    # "OK" | "FAIL"
    artifacts: Dict[str, Any]      # paths or small scalars
    notes: str = ""

def _ensure_registry():
    os.makedirs("data", exist_ok=True)
    if not REGISTRY_CSV.exists():
        with REGISTRY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ts","run_id","phase","status","artifacts","notes"])
            w.writeheader()

def register(record: RunRecord):
    _ensure_registry()
    ts = datetime.utcnow().isoformat()
    row = {
        "ts": ts,
        "run_id": record.run_id,
        "phase": record.phase,
        "status": record.status,
        "artifacts": json.dumps(record.artifacts, ensure_ascii=False),
        "notes": record.notes
    }
    with REGISTRY_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
