"""
Stage 2: How fast does a black hole scramble?

Stage 1 assumed a perfectly scrambled black hole (a random state). Here we
build the scrambling from real quantum gates, in BOTH Qiskit and CUDA-Q, and
ask: how many layers of gates does it take to reach the Page curve?

For each circuit depth we:
  1. draw random scrambling circuits (see scrambler.py),
  2. simulate each one in Qiskit and in CUDA-Q, and check they give the
     same quantum state (they should, to ~1e-15),
  3. compute the Page curve: radiation entropy after emitting k qubits.

The punchline: with nearest-neighbour "chain" connectivity, scrambling takes
many layers, because information has to ripple along the line. With
"all-to-all" connectivity it takes only a few. The "fast scrambling"
conjecture (Sekino & Susskind, 2008) says black holes are the all-to-all kind,
scrambling in a time that grows only like log(N).

Run:  .venv/bin/python stage2_scrambling.py
      .venv/bin/python stage2_scrambling.py --target nvidia   # on an NVIDIA GPU
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import scrambler_cudaq
import scrambler_qiskit
from page_formula import page_entropy
from scrambler import CONNECTIVITIES, make_scrambler, page_curve_of

# Colours (from a colour-blind-checked palette). Blue = chain, orange =
# all-to-all throughout; within a panel, darker means deeper circuit.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ORANGE_RAMP = ["#f3b08f", "#ec8a5c", "#d95926", "#b04416", "#7f2e0c"]
SERIES = {"chain": "#2a78d6", "all-to-all": "#eb6834"}
RAMPS = {"chain": BLUE_RAMP, "all-to-all": ORANGE_RAMP}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def run_experiment(n_qubits, max_depth, n_samples, seed):
    """
    Returns curves[connectivity][depth] = average Page curve (array of n+1),
    plus the worst Qiskit-vs-CUDA-Q fidelity and each framework's total time.
    """
    rng = np.random.default_rng(seed)
    curves = {c: {} for c in CONNECTIVITIES}
    worst_fidelity = 1.0
    seconds = {"Qiskit": 0.0, "CUDA-Q": 0.0}

    for connectivity in CONNECTIVITIES:
        for depth in range(1, max_depth + 1):
            samples = []
            for _ in range(n_samples):
                circuit = make_scrambler(n_qubits, depth, connectivity, rng)

                t0 = time.perf_counter()
                psi_qiskit = scrambler_qiskit.final_state(circuit)
                t1 = time.perf_counter()
                psi_cudaq = scrambler_cudaq.final_state(circuit)
                t2 = time.perf_counter()
                seconds["Qiskit"] += t1 - t0
                seconds["CUDA-Q"] += t2 - t1

                # Fidelity |<a|b>|^2 is 1 for identical states (it ignores an
                # overall phase, which has no physical meaning).
                fidelity = abs(np.vdot(psi_qiskit, psi_cudaq)) ** 2
                worst_fidelity = min(worst_fidelity, fidelity)

                samples.append(page_curve_of(psi_qiskit, n_qubits))
            curves[connectivity][depth] = np.mean(samples, axis=0)
    return curves, worst_fidelity, seconds


def style_axes(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot(curves, n_qubits, curve_depths, path):
    steps = np.arange(n_qubits + 1)
    page = [float(page_entropy(k, n_qubits)) for k in steps]
    half = n_qubits // 2
    max_depth = max(curves["chain"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    titles = {"chain": "Chain (neighbours only)",
              "all-to-all": "All-to-all (any pair)"}

    # Panels 1-2: Page curves at increasing depth, one panel per connectivity.
    for ax, connectivity in zip(axes[:2], CONNECTIVITIES):
        ax.plot(steps, page, "--", color=MUTED, linewidth=1.5,
                label="Fully scrambled (Page)")
        for depth, colour in zip(curve_depths, RAMPS[connectivity]):
            ax.plot(steps, curves[connectivity][depth], "-o", color=colour,
                    linewidth=2, markersize=4, label=f"{depth} layer{'s' * (depth > 1)}")
        ax.set_title(titles[connectivity], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Qubits emitted as radiation", color=MUTED)
        ax.set_ylim(0, n_qubits / 2 + 0.5)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        style_axes(ax)
    axes[0].set_ylabel("Radiation entropy (bits)", color=MUTED)

    # Panel 3: entropy at the Page time vs depth. How fast does each one scramble?
    ax = axes[2]
    depths = np.arange(1, max_depth + 1)
    ax.axhline(page[half], linestyle="--", color=MUTED, linewidth=1.5)
    ax.text(max_depth, page[half] + 0.12, "fully scrambled", color=MUTED,
            fontsize=8, ha="right")
    for connectivity in CONNECTIVITIES:
        values = [curves[connectivity][d][half] for d in depths]
        ax.plot(depths, values, "-o", color=SERIES[connectivity], linewidth=2,
                markersize=4, label=connectivity)
    ax.set_title(f"Entropy at the Page time ({half} of {n_qubits} emitted)",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Circuit depth (layers of gates)", color=MUTED)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(0, n_qubits / 2 + 0.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style_axes(ax)

    fig.suptitle("How fast does a toy black hole scramble?", color=INK,
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target", default=None,
                        help='CUDA-Q target, e.g. "qpp-cpu" or "nvidia" (GPU)')
    args = parser.parse_args()

    if args.target:
        scrambler_cudaq.set_target(args.target)
    print(f"CUDA-Q target: {scrambler_cudaq.target_name()}")
    print(f"Simulating {args.qubits}-qubit black holes, depths 1-{args.max_depth}, "
          f"{args.samples} random circuits each, in Qiskit and CUDA-Q...\n")

    curves, worst_fidelity, seconds = run_experiment(
        args.qubits, args.max_depth, args.samples, args.seed)

    half = args.qubits // 2
    target = float(page_entropy(half, args.qubits))
    print(f"Entropy at the Page time (fully scrambled = {target:.2f} bits):")
    print(f"{'depth':>6} {'chain':>8} {'all-to-all':>11}")
    for depth in range(1, args.max_depth + 1):
        print(f"{depth:>6} {curves['chain'][depth][half]:>8.2f} "
              f"{curves['all-to-all'][depth][half]:>11.2f}")

    for connectivity in CONNECTIVITIES:
        reached = [d for d in curves[connectivity]
                   if curves[connectivity][d][half] >= 0.95 * target]
        when = f"{reached[0]} layers" if reached else f"more than {args.max_depth} layers"
        print(f"{connectivity:>11} reaches 95% of fully scrambled after {when}")

    print(f"\nQiskit vs CUDA-Q: worst fidelity over all circuits = {worst_fidelity:.12f}")
    print(f"Time spent: Qiskit {seconds['Qiskit']:.1f}s, CUDA-Q {seconds['CUDA-Q']:.1f}s")

    plot(curves, args.qubits, curve_depths=[1, 4, 8, 16, 32], path="scrambling.png")
    print("Saved plot to scrambling.png")


if __name__ == "__main__":
    main()
