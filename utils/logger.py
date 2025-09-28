import csv, json, time, os, subprocess, sys
from pathlib import Path

class CSVLogger:
    def __init__(self, csv_path, fieldnames):
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.start = time.time()
        write_header = not os.path.exists(csv_path)
        self.f = open(csv_path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if write_header:
            self.w.writeheader()

    def log(self, row: dict):
        row = dict(row)
        row.setdefault("wall_time", time.time() - self.start)
        self.w.writerow(row); self.f.flush()

    def close(self):
        self.f.close()

def save_config(config: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 可选写入git版本
    try:
        commit = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
        config["git_commit"] = commit
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

class JSONLAggregator:
    """用于 ablations/ablation_log.jsonl 汇总每次实验的最终指标"""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "a")

    def log(self, summary: dict):
        self.f.write(json.dumps(summary) + "\n"); self.f.flush()

    def close(self):
        self.f.close()