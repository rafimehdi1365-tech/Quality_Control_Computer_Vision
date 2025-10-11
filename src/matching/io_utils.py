# src/matching/io_utils.py
import json
from pathlib import Path
from typing import Any, Dict, Iterable
import logging

def ensure_dir(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(record: Dict, path: Path):
    """
    Append one JSON object per line (for streaming debug).
    Safe to call concurrently-ish in GH Actions (simple append).
    """
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_jsonl(path: Path):
    out = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out
