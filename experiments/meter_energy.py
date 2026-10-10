#!/usr/bin/env python3
"""Measured energy per decision with an external USB or USB-C power meter.

Works on any machine that runs the harness: a Raspberry Pi with a USB meter between its
power supply and the board, or any MacBook (Intel or Apple Silicon) with a USB-C meter
between the charger and the laptop. The script uses the meter's accumulated-energy counter
(mWh). You read the counter, type it and press Enter; the script runs a predictor in a loop;
when the window ends it asks for the counter again. The loop keeps running until you press
Enter, so the time you take to read the meter is inside the window at both ends.

    energy per decision = (meter energy in the window - idle power x window) / decisions

On an Apple Silicon Mac where you can use sudo, the script also samples the chip's own power
telemetry (powermetrics) during the same windows, so the meter and the telemetry can be
compared on one machine.

Usage (from the artifact root):
    python experiments/meter_energy.py                         # 2 repetitions x 120 s
    python experiments/meter_energy.py --seconds 10 --reps 1   # rehearsal, about 5 minutes
    python experiments/meter_energy.py --onchip off            # meter only

Outputs (TAG is mac, pi or host unless --tag is given):
    results/meter_TAG_runs.csv, results/meter_TAG_summary.csv, results/meter_TAG_console.txt
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                      # single-threaded numerics, as in the paper

import argparse
import csv
import platform
import re
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from taxmeter.predictors import build_events, load_predictors  # noqa: E402

LOG = None
RE_ELAPSED = re.compile(r"\(([\d.]+)ms elapsed\)")
RE_CPU = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.M)


def say(text=""):
    print(text, flush=True)
    if LOG is not None:
        LOG.write(text + "\n")
        LOG.flush()


def ask_reading(prompt):
    while True:
        txt = input(prompt).strip().replace(",", ".")
        try:
            return float(txt)
        except ValueError:
            print("    please type a number such as 1234.56")


def onchip_power(path):
    """Mean CPU power (W) from a powermetrics text file, first sample dropped."""
    try:
        text = open(path).read()
    except OSError:
        return None
    pw, dt = [], []
    for b in text.split("*** Sampled system activity")[1:]:
        e, c = RE_ELAPSED.search(b), RE_CPU.search(b)
        if e and c:
            dt.append(float(e.group(1)))
            pw.append(float(c.group(1)) / 1000.0)
    pw, dt = pw[1:], dt[1:]
    if not pw:
        return None
    return sum(p * d for p, d in zip(pw, dt)) / sum(dt)


def window(seconds, work, label, onchip_path=None):
    """Run `work` (or sleep, if None) from the first Enter to the second Enter.
    Returns (mWh used, seconds elapsed, decisions made, on-chip W or None)."""
    start = ask_reading(f"\n[{label}] Type the meter's energy counter now (mWh), then press Enter: ")
    stop = threading.Event()
    state = {"calls": 0}

    def loop():
        if work is None:
            while not stop.is_set():
                time.sleep(0.05)
            return
        while not stop.is_set():
            for _ in range(200):
                work()
            state["calls"] += 200

    pm = pmf = None
    if onchip_path:
        subprocess.run(["sudo", "-n", "-v"], capture_output=True)
        pmf = open(onchip_path, "w")
        pm = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-i", "100",
                               "-n", str(max(3, int(seconds * 10)))], stdout=pmf, stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    while time.perf_counter() - t0 < seconds:
        left = seconds - (time.perf_counter() - t0)
        print(f"\r    running... {left:5.0f} s left   ", end="", flush=True)
        time.sleep(1.0)
    print("\r    time is up. Keep hands off until you have read the meter.")
    end = ask_reading(f"[{label}] Type the meter's energy counter now (mWh), then press Enter: ")
    elapsed = time.perf_counter() - t0
    stop.set()
    th.join()
    chip_w = None
    if pm is not None:
        try:
            pm.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pm.terminate()
        pmf.close()
        chip_w = onchip_power(onchip_path)
    if end < start:
        say("    the second reading is smaller than the first; this window will be repeated")
        return None
    if end - start < 5:
        say(f"    note: the counter moved only {end - start:.2f} mWh; if your meter shows whole mWh, "
            "rerun with --seconds 240 for better resolution")
    say(f"    [{label}] start {start:.3f} mWh, end {end:.3f} mWh, {elapsed:.1f} s"
        + (f", on-chip {chip_w:.3f} W" if chip_w is not None else ""))
    return end - start, elapsed, state["calls"], chip_w


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def machine_name(tag):
    """Short, LaTeX-safe processor or board name for the paper sentence."""
    name = ""
    if tag == "pi":
        try:
            m = re.search(r"Raspberry Pi \d", open("/proc/device-tree/model").read())
            name = m.group(0) if m else "Raspberry Pi"
        except OSError:
            name = "Raspberry Pi"
    elif sys.platform == "darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            name = ""
        name = re.sub(r"\((R|TM)\)", "", name)
        name = re.sub(r"\s*CPU\s*@.*$", "", name)
        name = re.sub(r"\s+", " ", name).strip()
    for a, b in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "")):
        name = name.replace(a, b)
    return name


def fmt_uj(x):
    return f"{x:.1f}" if x < 100 else f"{x:,.0f}"


def paper_sentence(tag, summ, r_paper, r_chip, ratio_lo, ratio_hi):
    """One sentence, in the paper's style, for the Measured energy paragraph of Section 5."""
    meas = [s["measured_uj_mean"] for s in summ]
    lo, hi = min(meas), max(meas)
    spread = hi / max(lo, 1e-9)
    name = machine_name(tag)
    if tag == "pi":
        return (f"With an external USB meter on a {name}, energy per decision ran from {fmt_uj(lo)} to "
                f"{fmt_uj(hi)}~\\uJ, a {spread:,.0f}\\texttimes{{}} spread, with a rank correlation of "
                f"{r_paper:.2f} against column B.")
    if r_chip is not None:
        return (f"On a second Mac with an {name or 'Apple Silicon chip'}, an external USB-C meter read "
                f"{ratio_lo:.1f} to {ratio_hi:.1f}\\texttimes{{}} the chip's telemetry in the same windows, as "
                f"expected for a meter that also sees memory, with rank correlations of {r_chip:.2f} against "
                f"the telemetry and {r_paper:.2f} against column B.")
    return (f"On a MacBook with an {name or 'Intel processor'}, an external USB-C meter measured a "
            f"{spread:,.0f}\\texttimes{{}} spread with a rank correlation of {r_paper:.2f} against column B.")


def detect_tag():
    if sys.platform == "darwin":
        return "mac"
    try:
        if "raspberry" in open("/proc/device-tree/model").read().lower():
            return "pi"
    except OSError:
        pass
    return "host"


def onchip_available():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0, help="length of each window")
    ap.add_argument("--reps", type=int, default=2, help="interleaved repetitions")
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--tag", default=None, help="label for the output files (default: mac, pi or host)")
    ap.add_argument("--onchip", choices=["auto", "on", "off"], default="auto",
                    help="also sample Apple Silicon power telemetry (needs sudo)")
    args = ap.parse_args()
    tag = args.tag or detect_tag()
    use_chip = (args.onchip == "on") or (args.onchip == "auto" and onchip_available())

    d = dict(np.load(os.path.join(ROOT, "data", "workload.npz")))
    events, _ = build_events(d)
    rng = np.random.default_rng(7)
    sample = [events[i] for i in rng.choice(len(events), size=min(args.events, len(events)), replace=False)]
    preds = load_predictors(os.path.join(ROOT, "weights"))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    raw_dir = os.path.join(ROOT, "results", f"meter_{tag}_powermetrics")
    if use_chip:
        os.makedirs(raw_dir, exist_ok=True)
    LOG = open(os.path.join(ROOT, "results", f"meter_{tag}_console.txt"), "w")
    total = (len(preds) + 1) * args.reps * (args.seconds + 20) / 60
    say(f"[meter_energy] {platform.platform()} | {platform.machine()} | {len(preds)} predictors | "
        f"{args.reps} reps x {args.seconds:.0f} s | about {total:.0f} min including meter readings | "
        f"on-chip telemetry {'on' if use_chip else 'off'}")
    say("Each window: read the meter's mWh counter, type it, press Enter; wait; read again, type, Enter.")

    rows = []
    for rep in range(args.reps):
        res = None
        while res is None:
            res = window(args.seconds, None, f"rep {rep + 1} idle",
                         os.path.join(raw_dir, f"idle_rep{rep}.txt") if use_chip else None)
        mwh, el, _, chip_idle = res
        idle_w = mwh * 3.6 / el
        say(f"    idle power {idle_w:.3f} W" + (f" (on-chip {chip_idle:.3f} W)" if chip_idle is not None else ""))
        order = preds if rep % 2 == 0 else list(reversed(preds))
        for p in order:
            state = {"i": 0}

            def one(p=p, state=state):
                ev = sample[state["i"]]
                state["i"] = (state["i"] + 1) % len(sample)
                p.predict(ev)
            for _ in range(2000):
                one()
            res = None
            while res is None:
                res = window(args.seconds, one, f"rep {rep + 1} {p.name}",
                             os.path.join(raw_dir, f"{p.name}_rep{rep}.txt") if use_chip else None)
            mwh, el, calls, chip_w = res
            active_w = mwh * 3.6 / el
            marg = max(active_w - idle_w, 0.0)
            e_uj = marg * el / calls * 1e6
            chip_uj = ""
            if chip_w is not None and chip_idle is not None:
                chip_uj = round(max(chip_w - chip_idle, 0.0) * el / calls * 1e6, 3)
            rows.append(dict(rep=rep, predictor=p.name, idle_w=round(idle_w, 4), active_w=round(active_w, 4),
                             marginal_w=round(marg, 4), window_s=round(el, 2), calls=calls,
                             mean_us=round(el / calls * 1e6, 3), measured_uj=round(e_uj, 3), mwh=round(mwh, 4),
                             onchip_uj=chip_uj))
            say(f"    {p.name:14s} {el / calls * 1e6:9.2f} us/decision  marginal {marg:6.3f} W  "
                f"meter {e_uj:9.2f} uJ" + (f"  on-chip {chip_uj:9.2f} uJ" if chip_uj != "" else ""))

    out = os.path.join(ROOT, "results", f"meter_{tag}_runs.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summ = []
    for p in preds:
        r = [x for x in rows if x["predictor"] == p.name]
        m = [x["measured_uj"] for x in r]
        c = [x["onchip_uj"] for x in r if x["onchip_uj"] != ""]
        summ.append(dict(predictor=p.name, measured_uj_mean=round(statistics.mean(m), 3),
                         measured_uj_sd=round(statistics.stdev(m), 3) if len(m) > 1 else 0.0,
                         marginal_w_mean=round(statistics.mean(x["marginal_w"] for x in r), 4),
                         mean_us=round(statistics.mean(x["mean_us"] for x in r), 3),
                         onchip_uj_mean=round(statistics.mean(c), 3) if c else ""))
    out2 = os.path.join(ROOT, "results", f"meter_{tag}_summary.csv")
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ[0].keys()); w.writeheader(); w.writerows(summ)

    meas = [s["measured_uj_mean"] for s in summ]
    say("\n[summary]")
    say(f"  measured energy spread across predictors: {max(meas) / max(min(meas), 1e-9):.0f}x")
    weak = [s["predictor"] for s in summ if s["marginal_w_mean"] < 0.05]
    if weak:
        say(f"  [warning] marginal power under 0.05 W for {', '.join(weak)}; rerun with --seconds 240")
    r_paper = r_chip = ratio_lo = ratio_hi = None
    pt = os.path.join(ROOT, "results", "paper_table1.csv")
    if os.path.exists(pt):
        paper = {r["predictor"]: float(r["E_uj_B"]) for r in csv.DictReader(open(pt))}
        common = [s for s in summ if s["predictor"] in paper]
        r_paper = spearman([s['measured_uj_mean'] for s in common], [paper[s['predictor']] for s in common])
        say(f"  rank correlation, meter vs paper column B ({len(common)} predictors): {r_paper:.2f}")
    both = [s for s in summ if s["onchip_uj_mean"] not in ("", 0, 0.0)]
    if both:
        ratios = sorted(s["measured_uj_mean"] / s["onchip_uj_mean"] for s in both)
        ratio_lo, ratio_hi = ratios[0], ratios[-1]
        r_chip = spearman([s['measured_uj_mean'] for s in both], [s['onchip_uj_mean'] for s in both])
        say(f"  meter / on-chip telemetry on this machine: {ratios[0]:.2f} to {ratios[-1]:.2f} "
            f"(median {statistics.median(ratios):.2f}); the meter also sees memory, storage and "
            f"power conversion, so a ratio above 1 is expected")
        say(f"  rank correlation, meter vs on-chip on this machine ({len(both)} predictors): "
            f"{spearman([s['measured_uj_mean'] for s in both], [s['onchip_uj_mean'] for s in both]):.2f}")
    say(f"  wrote {out}\n  wrote {out2}\n  wrote results/meter_{tag}_console.txt")
    if r_paper is not None and r_paper == r_paper:
        sentence = paper_sentence(tag, summ, r_paper, r_chip, ratio_lo, ratio_hi)
        with open(os.path.join(ROOT, "results", f"meter_{tag}_paper.tex"), "w") as f:
            f.write(sentence + "\n")
        say("\n[sentence for the paper, also saved as results/meter_" + tag + "_paper.tex]")
        say(sentence)
    LOG.close()


if __name__ == "__main__":
    main()
