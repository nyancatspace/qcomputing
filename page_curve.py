"""
Stage 1: The Page curve of an evaporating toy black hole.

THE STORY
---------
A black hole slowly evaporates by emitting Hawking radiation. The big question
(the "black hole information paradox") is whether the information about what
fell in is destroyed, or whether it leaks out, scrambled, in the radiation.

Don Page (1993) pointed out a test. Track the ENTANGLEMENT ENTROPY between the
radiation and the black hole that's left:

  * If information is LOST (Hawking's original calculation), every emitted
    particle is entangled with the black hole's interior, so the entropy of
    the radiation just keeps growing, even after the black hole is gone.
    That's a contradiction: nothing is left for the radiation to be
    entangled with.

  * If information is PRESERVED, the entropy rises, peaks halfway through
    evaporation (the "Page time"), then falls back to zero. The final
    radiation is a pure state that carries all the information.

THE TOY MODEL
-------------
  1. The black hole is N qubits.
  2. Black holes scramble information extremely well, so we model the
     black hole's state as a RANDOM quantum state of N qubits.
  3. "Emitting" k qubits of radiation means splitting the N qubits into two
     groups: k radiation qubits and N-k qubits still in the black hole.
  4. For each k, we compute the entanglement entropy between the groups.

Run:  .venv/bin/python page_curve.py
"""

import numpy as np
import matplotlib.pyplot as plt

N_QUBITS = 12   # size of our black hole. 2^12 = 4096 amplitudes, runs instantly.
N_SAMPLES = 20  # average over several random black holes to smooth the curve


def random_black_hole_state(n_qubits, rng):
    """
    A random ("Haar-random") pure state of n qubits.

    A quantum state of n qubits is a list of 2^n complex numbers (amplitudes)
    whose squared magnitudes add up to 1. Choosing each amplitude from a
    Gaussian distribution and then normalizing gives a state picked uniformly
    at random. That's our stand-in for a maximally scrambled black hole.
    """
    dim = 2 ** n_qubits
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


def entanglement_entropy(psi, k, n_qubits):
    """
    Entanglement entropy (in bits) between the first k qubits (the radiation)
    and the remaining n-k qubits (the black hole).

    How it works:
      * Reshape the 2^n amplitudes into a matrix with 2^k rows (radiation)
        and 2^(n-k) columns (black hole).
      * Its singular values s_i give the "Schmidt decomposition": the state
        written as a sum of paired radiation/black-hole states. The squares
        p_i = s_i^2 are probabilities.
      * Entropy S = -sum p_i log2 p_i.
        S = 0 means no entanglement: the two parts are independent.
        Each extra bit of S is like one more shared entangled pair.

    (Equivalently: the von Neumann entropy of the radiation's reduced density
    matrix, which we would get by "tracing out" the black hole.)
    """
    matrix = psi.reshape(2 ** k, 2 ** (n_qubits - k))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    p = singular_values ** 2
    p = p[p > 1e-15]  # drop zeros, since 0 * log(0) counts as 0
    return float(-np.sum(p * np.log2(p)))


def page_prediction(k, n_qubits):
    """
    Page's analytic formula for the average entropy of a random state.
    With m = 2^(smaller part) and n = 2^(larger part):  S ~ log2(m) - m/(2n ln 2)
    """
    small, large = sorted((k, n_qubits - k))
    m, n = 2 ** small, 2 ** large
    return small - m / (2 * n * np.log(2))


def main():
    rng = np.random.default_rng(seed=42)
    steps = np.arange(N_QUBITS + 1)  # k = 0, 1, ..., N qubits emitted

    # Simulate evaporation for many random black holes and average.
    entropies = np.zeros((N_SAMPLES, len(steps)))
    for s in range(N_SAMPLES):
        psi = random_black_hole_state(N_QUBITS, rng)
        for k in steps:
            entropies[s, k] = entanglement_entropy(psi, k, N_QUBITS)
    simulated = entropies.mean(axis=0)

    theory = [page_prediction(k, N_QUBITS) for k in steps]
    hawking = steps  # information-loss picture: +1 bit per emitted qubit, forever

    print(f"{'emitted':>8} {'simulated':>10} {'Page theory':>12} {'Hawking':>8}")
    for k in steps:
        print(f"{k:>8} {simulated[k]:>10.3f} {theory[k]:>12.3f} {hawking[k]:>8}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, hawking, "--", color="gray",
             label="Hawking (information lost)")
    plt.plot(steps, theory, "-", color="tab:blue", alpha=0.6,
             label="Page's formula (information preserved)")
    plt.plot(steps, simulated, "o", color="tab:red",
             label=f"Simulation ({N_SAMPLES} random {N_QUBITS}-qubit black holes)")
    plt.axvline(N_QUBITS / 2, color="black", linestyle=":", linewidth=1)
    plt.text(N_QUBITS / 2 + 0.15, 0.3, "Page time", fontsize=9)
    plt.xlabel("Qubits emitted as Hawking radiation")
    plt.ylabel("Entanglement entropy of radiation (bits)")
    plt.title("The Page curve of a toy black hole")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("page_curve.png", dpi=150)
    print("\nSaved plot to page_curve.png")


if __name__ == "__main__":
    main()
