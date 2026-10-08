"""
The circuit recipe for a scrambling black hole, independent of any framework.

Page's formula assumes a perfectly scrambled (random) state. Here
we build the scrambling out of actual quantum gates, the way a real quantum
computer would have to, and watch it happen layer by layer.

One LAYER of the circuit is:
  1. A random rotation on every qubit (three angles each: Rz, Ry, Rz).
     This mixes up each qubit on its own.
  2. A CZ gate on pairs of qubits. This is what creates entanglement:
     information can only spread from one qubit to another through these.

Which qubits get paired is the "connectivity", and it turns out to matter a lot:

  * "chain": qubits sit in a line and only neighbours interact, in a
    brick-wall pattern: pairs (0,1),(2,3),... then (1,2),(3,4),...
    Information spreads like a ripple, one step per layer.

  * "all-to-all": each layer pairs qubits up at random, so any qubit can
    talk to any other. Physicists conjecture black holes work like this,
    which would make them the fastest possible scramblers in nature.

The same recipe is turned into a Qiskit circuit (scrambler_qiskit.py) and a
CUDA-Q kernel (scrambler_cudaq.py), so the two frameworks run identical gates.
"""

from dataclasses import dataclass

import numpy as np

CONNECTIVITIES = ("chain", "all-to-all")


@dataclass
class Scrambler:
    n_qubits: int
    angles: np.ndarray            # shape (depth, n_qubits, 3): Rz, Ry, Rz angles
    pairs: list[list[tuple[int, int]]]  # CZ pairs for each layer

    @property
    def depth(self):
        return len(self.pairs)

    def flat_args(self):
        """
        The same recipe as plain flat lists, because CUDA-Q kernels take simple
        arguments (ints, lists of floats or ints) rather than Python objects.
        Layer L's CZ pairs are pairs number layer_starts[L] .. layer_starts[L+1]-1,
        and pair p acts on qubits pair_qubits[2p] and pair_qubits[2p+1].
        """
        pair_qubits, layer_starts = [], [0]
        for layer in self.pairs:
            for a, b in layer:
                pair_qubits += [a, b]
            layer_starts.append(len(pair_qubits) // 2)
        return self.angles.ravel().tolist(), pair_qubits, layer_starts


def make_scrambler(n_qubits, depth, connectivity, rng):
    """Draw a random scrambling circuit with the given number of layers."""
    if connectivity not in CONNECTIVITIES:
        raise ValueError(f"connectivity must be one of {CONNECTIVITIES}")

    angles = rng.uniform(0, 2 * np.pi, size=(depth, n_qubits, 3))
    pairs = []
    for layer in range(depth):
        if connectivity == "chain":
            start = layer % 2  # alternate the bricks: (0,1),(2,3).. then (1,2),(3,4)..
            order = range(start, n_qubits - 1, 2)
            pairs.append([(i, i + 1) for i in order])
        else:
            shuffled = rng.permutation(n_qubits)
            pairs.append([(int(shuffled[i]), int(shuffled[i + 1]))
                          for i in range(0, n_qubits - 1, 2)])
    return Scrambler(n_qubits, angles, pairs)


def radiation_entropy(psi, k, n_qubits):
    """
    Entanglement entropy (in bits) between radiation qubits 0..k-1 and the
    black-hole qubits k..n-1.

    Both Qiskit and CUDA-Q store states "little-endian": qubit 0 is the
    lowest bit of the amplitude index. So qubits 0..k-1 are the last k bits,
    meaning the columns when we reshape into a 2^(n-k) x 2^k matrix.
    """
    matrix = np.asarray(psi).reshape(2 ** (n_qubits - k), 2 ** k)
    p = np.linalg.svd(matrix, compute_uv=False) ** 2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def page_curve_of(psi, n_qubits):
    """Entropy after emitting k = 0, 1, ..., n qubits."""
    return np.array([radiation_entropy(psi, k, n_qubits) for k in range(n_qubits + 1)])
