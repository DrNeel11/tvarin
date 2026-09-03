"""Phase 2 benchmark: characterize the simulated remote-memory link itself,
before any model is involved.

Two sweeps:
  A. Latency sweep at various block sizes, bandwidth unthrottled — shows
     when small-block transfers are latency-dominated vs. when large blocks
     start to matter more than the per-request delay.
  B. Bandwidth sweep at a fixed representative latency, various block sizes
     — shows the regime where bandwidth, not latency, is the bottleneck.

Block sizes are chosen to bracket realistic single-transformer-layer sizes:
a Qwen2.5-0.5B layer is roughly 15-40MB (bf16); a 7B layer is roughly
400-550MB (bf16). 1/16/64/256 MB spans that range.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memtier.config import RESULTS_DIR
from memtier.remote.client import RemoteMemoryClient
from memtier.remote.server import RemoteMemoryServer

LATENCIES_MS = [1, 5, 10, 25, 50]
BANDWIDTHS_MB_PER_S = [1000, 100, 25, 10, 2]  # ~LAN, fast-home, mid-home, slow-home, poor
SIZES_MB = [1, 16, 64, 256]
N_REPS = 5
FIXED_LATENCY_FOR_BANDWIDTH_SWEEP_MS = 10


def start_server(latency_ms: float, bandwidth_mb_per_s, port: int, log_path: Path):
    server = RemoteMemoryServer(
        host="127.0.0.1", port=port, latency_ms=latency_ms,
        bandwidth_mb_per_s=bandwidth_mb_per_s, log_path=log_path,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    return server


def run_one_condition(port, latency_ms, bandwidth_mb_per_s, log_path, sizes_mb, out_rows, condition_label):
    server = start_server(latency_ms, bandwidth_mb_per_s, port, log_path)
    try:
        with RemoteMemoryClient(host="127.0.0.1", port=port, upload_bandwidth_mb_per_s=bandwidth_mb_per_s) as client:
            for size_mb in sizes_mb:
                key = f"block_{size_mb}mb"
                payload = os.urandom(size_mb * 1_000_000)

                put_result = client.put(key, payload)

                get_times = []
                for _ in range(N_REPS):
                    r = client.get(key)
                    assert r.found
                    get_times.append(r.total_s)
                get_times.sort()
                median_get_s = get_times[len(get_times) // 2]
                throughput_mb_per_s = (size_mb / median_get_s) if median_get_s > 0 else None

                row = dict(
                    condition=condition_label,
                    latency_ms=latency_ms,
                    bandwidth_mb_per_s=bandwidth_mb_per_s,
                    size_mb=size_mb,
                    put_s=put_result.total_s,
                    median_get_s=median_get_s,
                    min_get_s=get_times[0],
                    max_get_s=get_times[-1],
                    effective_throughput_mb_per_s=throughput_mb_per_s,
                )
                out_rows.append(row)
                throughput_str = f"{throughput_mb_per_s:.1f}MB/s" if throughput_mb_per_s else "n/a"
                print(
                    f"[{condition_label}] latency={latency_ms}ms bw={bandwidth_mb_per_s}"
                    f" size={size_mb}MB -> get={median_get_s*1000:.1f}ms"
                    f" throughput={throughput_str}"
                )
    finally:
        server.stop()
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="phase2_remote_link")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{ts}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    port = 5960

    print("=== Sweep A: latency, bandwidth unthrottled ===")
    for latency_ms in LATENCIES_MS:
        run_one_condition(
            port, latency_ms, None, out_dir / f"server_lat{latency_ms}.jsonl",
            SIZES_MB, rows, "latency_sweep",
        )
        port += 1

    print("\n=== Sweep B: bandwidth, latency fixed at "
          f"{FIXED_LATENCY_FOR_BANDWIDTH_SWEEP_MS}ms ===")
    for bw in BANDWIDTHS_MB_PER_S:
        run_one_condition(
            port, FIXED_LATENCY_FOR_BANDWIDTH_SWEEP_MS, bw,
            out_dir / f"server_bw{bw}.jsonl", SIZES_MB, rows, "bandwidth_sweep",
        )
        port += 1

    with open(out_dir / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
