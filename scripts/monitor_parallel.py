#!/usr/bin/env python3
import sys, time, os, math
from datetime import timedelta

def parse_joblog(p):
    done = 0
    runtimes = []
    if not os.path.exists(p): return 0,0,0.0
    with open(p, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # skip header if present
    for ln in lines[1:] if lines and lines[0].startswith('Seq') else lines:
        parts = ln.split('\t')
        # columns: Seq Starttime Runtime Send Receive ExitSignal ExitCode Host Command
        try:
            rt = float(parts[2]); exitcode = int(parts[7])
            if exitcode == 0:
                done += 1; runtimes.append(rt)
        except Exception:
            pass
    avg = sum(runtimes)/len(runtimes) if runtimes else 0.0
    return done, len(lines) - 1 if lines and lines[0].startswith('Seq') else len(lines), avg

if __name__ == "__main__":
    joblog = sys.argv[1]
    total = int(sys.argv[2])  # total jobs planned (== lines in manifest)
    done, seen, avg = parse_joblog(joblog)
    remain = max(0, total - done)
    eta_sec = remain * avg
    print(f"Jobs done: {done}/{total}  (seen:{seen})  avg sec/job: {avg:.1f}  ETA: {timedelta(seconds=math.ceil(eta_sec))}")
