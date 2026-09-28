#!/usr/bin/env python3
# CUDALM v0.7 Phase A — kernel-family aggregation over the Nsight Systems
# "CUDA GPU Kernel Summary" CSV (nsys stats --report cuda_gpu_kern_sum
# --format csv).
#
# Usage:
#   aggregate_kernel_families.py <kern_sum.csv> <completed_traversals>
#
# Emits (stdout, appended to benchmarks/profiling/v07_kernel_families.txt):
#   * per-family table:  family | total GPU time | % kernel time | calls |
#     avg latency
#   * top 10 kernels by total GPU time
#   * top 10 kernels by invocation count
#   * short-duration kernel share (avg duration < 100 us)
#   * kernels per completed model traversal
#
# Families (matching order matters: first match wins):
#   w4a16/int4 projection  : int4gemv
#   bf16 lm head           : bf16_gemv
#   deltanet               : deltanet (conv/gbeta/delta/gated_rmsnorm)
#   paged attention        : paged | attention | kv_write
#   rope                   : rope
#   embedding              : embed
#   rmsnorm                : rmsnorm
#   elementwise/residual/swiglu: add | silu | gate_mul | split_q_gate
#   other                  : anything else (incl. unexpected names)

import csv
import sys

SHORT_US = 100.0  # "short-duration" threshold for the fragmentation analysis


def family_of(name: str) -> str:
    n = name.lower()
    if "int4gemv" in n:
        return "w4a16/int4 projection"
    if "bf16_gemv" in n:
        return "bf16 lm head"
    if "deltanet" in n:
        return "deltanet"
    if "paged" in n or "attention" in n or "kv_write" in n:
        return "paged attention"
    if "rope" in n:
        return "rope"
    if "embed" in n:
        return "embedding"
    if "rmsnorm" in n:
        return "rmsnorm"
    if "add" in n or "silu" in n or "gate_mul" in n or "split_q_gate" in n:
        return "elementwise/residual/swiglu"
    return "other"


def parse_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            if header is None:
                # first non-empty row is the header
                header = [c.strip().lower() for c in row]
                continue
            rows.append(row)
    if header is None:
        raise SystemExit("empty CSV: " + path)
    idx = {}
    for i, h in enumerate(header):
        if "kernel name" in h or h == "name" or h == "operation":
            idx["name"] = i
        elif "total time" in h:
            idx["total"] = i
        elif h.startswith("instances") or h == "count" or h == "num calls":
            idx["count"] = i
        elif h.startswith("avg"):
            idx["avg"] = i
    for key in ("name", "total", "count"):
        if key not in idx:
            raise SystemExit("missing column %r in header %r" % (key, header))
    out = []
    for row in rows:
        try:
            name = row[idx["name"]]
            total = float(row[idx["total"]])
            count = int(float(row[idx["count"]]))
        except (ValueError, IndexError):
            continue
        avg = None
        if "avg" in idx:
            try:
                avg = float(row[idx["avg"]])
            except ValueError:
                avg = None
        out.append((name, total, count, avg))
    return out, header


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    traversals = 0
    if len(sys.argv) > 2:
        try:
            traversals = int(sys.argv[2])
        except ValueError:
            traversals = 0
    rows, header = parse_csv(path)
    # nsys 2022.x reports "Total Time (ns)" and "Avg (ns)" in the CSV; detect
    # the unit from the header and normalize to ns.
    unit = "ns"
    for h in header:
        if "total time" in h:
            if "us" in h:
                unit = "us"
            elif "ms" in h:
                unit = "ms"
            break
    factor = {"ns": 1.0, "us": 1e3, "ms": 1e6}[unit]

    total_time_ns = 0.0
    total_calls = 0
    fam = {}
    for name, total, count, avg in rows:
        total_ns = total * factor
        total_time_ns += total_ns
        total_calls += count
        f = family_of(name)
        agg = fam.setdefault(f, [0.0, 0])
        agg[0] += total_ns
        agg[1] += count

    print()
    print("units: per-kernel numbers in the CSV were %s; normalized to ns here"
          % unit)
    print()
    print("== kernel-family table (sorted by total GPU time) ==")
    print("family | total GPU time (ms) | %% kernel time | calls | avg latency (us/call)")
    for f, (t_ns, c) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        avg_us = (t_ns / c / 1e3) if c else 0.0
        pct = 100.0 * t_ns / total_time_ns if total_time_ns else 0.0
        print("%s | %.3f | %.2f%% | %d | %.3f" % (f, t_ns / 1e6, pct, c, avg_us))
    print()
    print("TOTAL: %.3f ms over %d kernel launches" % (total_time_ns / 1e6, total_calls))

    print()
    print("== top 10 kernels by TOTAL GPU time ==")
    print("kernel | family | total (ms) | %% of kernel time | calls | avg (us)")
    for name, total, count, avg in sorted(rows, key=lambda r: -r[1])[:10]:
        t_ns = total * factor
        avg_us = (avg * factor / 1e3) if avg is not None else (t_ns / count / 1e3 if count else 0.0)
        print("%s | %s | %.3f | %.2f%% | %d | %.3f"
              % (name, family_of(name), t_ns / 1e6,
                 100.0 * t_ns / total_time_ns if total_time_ns else 0.0,
                 count, avg_us))

    print()
    print("== top 10 kernels by INVOCATION count ==")
    print("kernel | family | calls | total (ms) | avg (us)")
    for name, total, count, avg in sorted(rows, key=lambda r: -r[2])[:10]:
        t_ns = total * factor
        avg_us = (avg * factor / 1e3) if avg is not None else (t_ns / count / 1e3 if count else 0.0)
        print("%s | %s | %d | %.3f | %.3f"
              % (name, family_of(name), count, t_ns / 1e6, avg_us))

    # short-duration share (fragmentation signal): kernels whose AVG duration
    # is < SHORT_US
    short_calls = 0
    short_time_ns = 0.0
    short_kinds = 0
    for name, total, count, avg in rows:
        t_ns = total * factor
        avg_us = (avg * factor / 1e3) if avg is not None else (t_ns / count / 1e3 if count else 0.0)
        if avg_us < SHORT_US:
            short_calls += count
            short_time_ns += t_ns
            short_kinds += 1
    print()
    print("== short-duration kernels (avg < %.0f us) ==" % SHORT_US)
    print("kinds: %d/%d | launches: %d/%d (%.2f%%) | time: %.3f ms (%.2f%% of kernel time)"
          % (short_kinds, len(rows), short_calls, total_calls,
             100.0 * short_calls / total_calls if total_calls else 0.0,
             short_time_ns / 1e6,
             100.0 * short_time_ns / total_time_ns if total_time_ns else 0.0))

    print()
    print("== kernels per completed model traversal ==")
    if traversals > 0:
        print("completed traversals (whole profiled process, both modes): %d" % traversals)
        print("total kernel launches: %d" % total_calls)
        print("kernels per traversal: %.2f" % (total_calls / traversals))
    else:
        print("traversal count not provided; launches = %d" % total_calls)


if __name__ == "__main__":
    main()
