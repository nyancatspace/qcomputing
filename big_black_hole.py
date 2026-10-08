"""
Stage 2b: A 2048-qubit black hole.

Two things in one script:

PART A - Page's exact formula, evaluated for big black holes.
    Page's (1993) exact result for the average entanglement entropy between
    a system of dimension m and one of dimension n (m <= n), in a random state:

        S = [ 1/(n+1) + 1/(n+2) + ... + 1/(mn) ]  -  (m - 1) / (2n)     (nats)

    For N qubits with k emitted, m = 2^min(k, N-k) and n = 2^max(k, N-k).
    The sum has up to 2^N terms, so we use the identity
        1/(n+1) + ... + 1/(mn) = digamma(mn + 1) - digamma(n + 1)
    and evaluate it with mpmath, which handles numbers like 2^(10^77) fine.

PART B - An actual simulation of a 2048-qubit scrambling black hole, using
    Clifford circuits (see clifford_scrambler.py), compared with Part A.
    Then the "fast scrambling" test: how does the number of layers needed
    to scramble grow as the black hole gets bigger?

Run:  .venv/bin/python big_black_hole.py              (about 5 minutes)
      .venv/bin/python big_black_hole.py --plot-only  (redraw from saved results)
"""

import argparse
import json
import time

import matplotlib.pyplot as plt
import mpmath
import numpy as np

from clifford_scrambler import CliffordBlackHole

mpmath.mp.dps = 120  # digits of precision: enough to resolve O(1) bits on top of 10^77

N_BIG = 2048
SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
RESULTS_FILE = "big_black_hole_results.json"

BLUE, ORANGE = "#2a78d6", "#eb6834"  # chain, all-to-all (same as stage 2)
INK, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#e1e0d9", "#a8a69f"


# ---------------------------------------------------------------- Part A

def page_entropy_exact(k, n_qubits):
    """Page's exact average entropy, in bits, after emitting k of n_qubits qubits."""
    small, large = sorted((k, n_qubits - k))
    m = mpmath.mpf(2) ** small
    n = mpmath.mpf(2) ** large
    nats = mpmath.digamma(m * n + 1) - mpmath.digamma(n + 1) - (m - 1) / (2 * n)
    return nats / mpmath.log(2)


def part_a():
    print("PART A: Page's exact formula\n")
    print(f"{N_BIG}-qubit black hole:")
    print(f"{'emitted k':>10} {'max possible':>13} {'Page entropy (bits)':>22} {'shortfall':>10}")
    for k in (1, 10, 512, 1014, 1020, 1023, 1024, 1025, 1028, 1536, 2047):
        s = page_entropy_exact(k, N_BIG)
        cap = min(k, N_BIG - k)
        shortfall = cap - s if cap - s > 1e-100 else 0  # hide rounding noise
        print(f"{k:>10} {cap:>13} {mpmath.nstr(s, 12):>22} {mpmath.nstr(shortfall, 3):>10}")

    # A black hole with the mass of the Sun has a Bekenstein-Hawking entropy of
    # about 10^77 bits, which we can treat as N "qubits".
    n_sun = 10 ** 77
    s_half = page_entropy_exact(n_sun // 2, n_sun)
    print(f"\nSun-mass black hole, N = 10^77 qubits, at the Page time (half emitted):")
    print(f"  Page entropy = N/2 - {mpmath.nstr(n_sun // 2 - s_half, 6)} bits")
    print("  At ANY size the curve is the triangle min(k, N-k), short by under 0.73 bits.\n")

    curve = [float(page_entropy_exact(k, N_BIG)) for k in range(N_BIG + 1)]
    return curve


# ---------------------------------------------------------------- Part B

def scramble(n_qubits, connectivity, seed, max_depth, check_every, stop_when_scrambled):
    """
    Apply layers until the entropy at the Page time is within 2 bits of its
    maximum (n/2). Returns the depth that took (None if never), the
    trajectory [(depth, S_mid)], and the final Page curve.
    """
    bh = CliffordBlackHole(n_qubits, connectivity, seed)
    half = n_qubits // 2
    trajectory, scrambled_at, curve = [], None, None
    while bh.depth < max_depth:
        bh.apply_layer()
        if bh.depth % check_every and bh.depth != max_depth:
            continue
        curve = bh.page_curve()
        trajectory.append((bh.depth, int(curve[half])))
        if scrambled_at is None and curve[half] >= half - 2:
            scrambled_at = bh.depth
            if stop_when_scrambled:
                break
    return scrambled_at, trajectory, curve.tolist()


def part_b():
    print(f"PART B: Clifford simulation\n")
    t0 = time.time()

    # The big runs: follow both connectivities at 2048 qubits all the way.
    big = {}
    for connectivity, max_depth, every in (("all-to-all", 40, 1), ("chain", 4096, 32)):
        depth, trajectory, curve = scramble(N_BIG, connectivity, seed=1, max_depth=max_depth,
                                            check_every=every, stop_when_scrambled=False)
        big[connectivity] = {"scrambled_at": depth, "trajectory": trajectory, "curve": curve}
        print(f"  {N_BIG} qubits, {connectivity:>10}: scrambled after {depth} layers "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    # Scaling: scrambling depth vs size. Small black holes fluctuate more, so
    # average over more random circuits for them.
    scaling = {"all-to-all": {}, "chain": {}}
    for n in SIZES:
        seeds = range(10, 10 + max(1, min(8, 2048 // n)))
        for connectivity in scaling:
            if n == N_BIG:
                depths = [big[connectivity]["scrambled_at"]]
            else:
                every = 1 if connectivity == "all-to-all" else max(1, n // 64)
                max_depth = 60 if connectivity == "all-to-all" else 4 * n
                depths = [scramble(n, connectivity, s, max_depth, every, True)[0] for s in seeds]
            scaling[connectivity][n] = float(np.mean(depths))
        print(f"  {n:>5} qubits: all-to-all {scaling['all-to-all'][n]:6.1f} layers, "
              f"chain {scaling['chain'][n]:7.1f} layers ({time.time() - t0:.0f}s elapsed)",
              flush=True)
    return big, scaling


# ---------------------------------------------------------------- plotting

def style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot(results, path):
    page = np.array(results["page_curve"])
    sim = np.array(results["big"]["all-to-all"]["curve"])
    k = np.arange(N_BIG + 1)
    half = N_BIG // 2

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    (ax_full, ax_zoom), (ax_time, ax_scale) = axes

    # (a) The whole Page curve at 2048 qubits.
    ax_full.plot(k, k, ":", color=LIGHT, linewidth=1.5, label="Hawking (information lost)")
    ax_full.plot(k, page, "-", color=MUTED, linewidth=4, alpha=0.35,
                 label="Page's exact formula")
    ax_full.plot(k, sim, "-", color=ORANGE, linewidth=1.5,
                 label="Simulation (all-to-all, 40 layers)")
    ax_full.set_ylim(0, N_BIG * 0.62)
    ax_full.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax_full, f"The Page curve of a {N_BIG}-qubit black hole",
          "Qubits emitted as radiation", "Radiation entropy (bits)")

    # (b) Zoom in on the peak, where the curve falls short of the triangle.
    window = slice(half - 12, half + 13)
    ax_zoom.plot(k[window], np.minimum(k, N_BIG - k)[window], ":", color=LIGHT,
                 linewidth=1.5, label="Maximum possible, min(k, N-k)")
    ax_zoom.plot(k[window], page[window], "-", color=MUTED, linewidth=2,
                 label="Page's exact formula")
    ax_zoom.plot(k[window], sim[window], "o", color=ORANGE, markersize=5,
                 label="Simulation (whole bits only)")
    ax_zoom.annotate(f"{page[half]:.2f}", (half, page[half]), xytext=(0, -30),
                     textcoords="offset points", color=MUTED, fontsize=9, ha="center")
    ax_zoom.legend(fontsize=8, frameon=False, loc="lower center")
    style(ax_zoom, "Zoomed in on the Page time", "Qubits emitted as radiation")

    # (c) Scrambling over time at 2048 qubits.
    for connectivity, colour in (("chain", BLUE), ("all-to-all", ORANGE)):
        d, s = zip(*results["big"][connectivity]["trajectory"])
        ax_time.plot(d, np.array(s) / half * 100, "-", color=colour, linewidth=2,
                     label=connectivity)
    ax_time.set_xscale("log")
    ax_time.set_ylim(0, 105)
    ax_time.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax_time, f"Scrambling {N_BIG} qubits", "Circuit depth (layers of gates, log scale)",
          "Entropy at Page time (% of maximum)")

    # (d) Scrambling depth vs size.
    for connectivity, colour in (("chain", BLUE), ("all-to-all", ORANGE)):
        sizes = [int(n) for n in results["scaling"][connectivity]]
        depths = list(results["scaling"][connectivity].values())
        ax_scale.plot(sizes, depths, "-o", color=colour, linewidth=2, markersize=5,
                      label=connectivity)
        ax_scale.annotate(f"{depths[-1]:.0f}", (sizes[-1], depths[-1]), xytext=(6, -3),
                          textcoords="offset points", color=MUTED, fontsize=9)
    ax_scale.set_xscale("log", base=2)
    ax_scale.set_yscale("log")
    ax_scale.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax_scale, "Layers needed to scramble vs black hole size",
          "Number of qubits N (log scale)", "Layers to scramble (log scale)")

    fig.suptitle(f"A {N_BIG}-qubit toy black hole", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help=f"redraw the plot from {RESULTS_FILE}")
    args = parser.parse_args()

    if args.plot_only:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        page_curve = part_a()
        big, scaling = part_b()
        results = {"page_curve": page_curve, "big": big, "scaling": scaling}
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f)

        sim_mid = results["big"]["all-to-all"]["curve"][N_BIG // 2]
        print(f"\nAt the Page time: Page's formula {page_curve[N_BIG // 2]:.3f} bits, "
              f"simulation {sim_mid} bits (max {N_BIG // 2})")

    plot(results, "big_black_hole.png")
    print("Saved plot to big_black_hole.png")


if __name__ == "__main__":
    main()
