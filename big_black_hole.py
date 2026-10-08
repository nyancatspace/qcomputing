"""
The Page curve of a 2048-qubit toy black hole.

THE STORY
---------
A black hole slowly evaporates by emitting Hawking radiation. Does the
information that fell in get destroyed, or does it leak out in the radiation?
Don Page (1993) showed how to tell: track the ENTANGLEMENT ENTROPY between
the radiation and what's left of the black hole.

  * If information is LOST (Hawking's original picture), the entropy just
    keeps rising, one bit per emitted qubit.
  * If information is PRESERVED, it rises, peaks halfway (the "Page time"),
    then falls back to zero. That rise-and-fall is the Page curve.

WHAT THIS SCRIPT DOES
---------------------
  1. Simulates a 2048-qubit black hole: start with all qubits in |0>, then
     scramble them with layers of random Clifford gates (clifford_scrambler.py).
  2. "Emits" qubits one at a time and measures the radiation's entropy after
     each, giving the whole Page curve.
  3. Compares it with Page's exact formula:

        S = [ 1/(n+1) + 1/(n+2) + ... + 1/(mn) ]  -  (m - 1) / (2n)     (nats)

     where m = 2^(smaller part) and n = 2^(bigger part). The sum has up to
     2^2048 terms, so we use the digamma-function shortcut
        1/(n+1) + ... + 1/(mn) = digamma(mn + 1) - digamma(n + 1)
     and high-precision arithmetic from mpmath.

Run:  .venv/bin/python big_black_hole.py      (takes about 15 seconds)
"""

import matplotlib.pyplot as plt
import mpmath
import numpy as np

from clifford_scrambler import CliffordBlackHole

N_QUBITS = 2048
LAYERS = 20  # about 11 layers already scramble 2048 qubits; 20 to be safe

mpmath.mp.dps = 50  # work with 50 significant digits

ORANGE = "#eb6834"
INK, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#e1e0d9", "#a8a69f"


def page_entropy(k, n_qubits):
    """Page's exact average entropy, in bits, after emitting k of n_qubits qubits."""
    small, big = sorted((k, n_qubits - k))
    m = mpmath.mpf(2) ** small
    n = mpmath.mpf(2) ** big
    nats = mpmath.digamma(m * n + 1) - mpmath.digamma(n + 1) - (m - 1) / (2 * n)
    return float(nats / mpmath.log(2))


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


def plot(simulated, formula, path):
    k = np.arange(N_QUBITS + 1)
    half = N_QUBITS // 2
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: the whole evaporation.
    ax_full.plot(k, k, ":", color=LIGHT, linewidth=1.5, label="Hawking (information lost)")
    ax_full.plot(k, formula, "-", color=MUTED, linewidth=4, alpha=0.35,
                 label="Page's exact formula")
    ax_full.plot(k, simulated, "-", color=ORANGE, linewidth=1.5,
                 label=f"Simulation ({LAYERS} layers of gates)")
    ax_full.set_ylim(0, N_QUBITS * 0.62)
    ax_full.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax_full, f"The Page curve of a {N_QUBITS}-qubit black hole",
          "Qubits emitted as radiation", "Radiation entropy (bits)")

    # Right: zoom in on the peak, where the curve falls just short of the triangle.
    w = slice(half - 12, half + 13)
    ax_zoom.plot(k[w], np.minimum(k, N_QUBITS - k)[w], ":", color=LIGHT, linewidth=1.5,
                 label="Maximum possible, min(k, N-k)")
    ax_zoom.plot(k[w], formula[w], "-", color=MUTED, linewidth=2, label="Page's exact formula")
    ax_zoom.plot(k[w], simulated[w], "o", color=ORANGE, markersize=5,
                 label="Simulation (whole bits only)")
    ax_zoom.annotate(f"{formula[half]:.2f}", (half, formula[half]), xytext=(0, -30),
                     textcoords="offset points", color=MUTED, fontsize=9, ha="center")
    ax_zoom.legend(fontsize=8, frameon=False, loc="lower center")
    style(ax_zoom, "Zoomed in on the Page time", "Qubits emitted as radiation")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main():
    print(f"Scrambling a {N_QUBITS}-qubit black hole with {LAYERS} layers of gates...")
    black_hole = CliffordBlackHole(N_QUBITS, seed=1)
    for _ in range(LAYERS):
        black_hole.apply_layer()

    print("Measuring the radiation's entropy after each emitted qubit...")
    simulated = black_hole.page_curve()

    print("Evaluating Page's exact formula...\n")
    formula = np.array([page_entropy(k, N_QUBITS) for k in range(N_QUBITS + 1)])

    half = N_QUBITS // 2
    print(f"{'emitted k':>10} {'simulation':>11} {'Page formula':>13} {'max possible':>13}")
    for k in (0, 1, 512, half - 3, half - 1, half, half + 1, half + 3, 1536, N_QUBITS - 1, N_QUBITS):
        print(f"{k:>10} {simulated[k]:>11} {formula[k]:>13.3f} {min(k, N_QUBITS - k):>13}")

    plot(simulated, formula, "big_black_hole.png")
    print("\nSaved plot to big_black_hole.png")


if __name__ == "__main__":
    main()
