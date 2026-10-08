"""
Scrambling circuits for BIG black holes (thousands of qubits) using Clifford gates.

WHY A DIFFERENT METHOD?
-----------------------
The direct way to simulate qubits stores the full quantum state: 2^N complex
amplitudes. That's fine for 12 qubits (4,096 numbers) but hopeless for 2048 qubits: 2^2048 is about
10^616, far more than the number of atoms in the universe.

There's a famous loophole, the Gottesman-Knill theorem. If every gate is a
CLIFFORD gate (built from H, S and CNOT), the state can be described
by just N "stabilizers": N Pauli strings like X Z I Y ... that the state is
unchanged by. Each stabilizer is 2N bits, so the whole state is an N x 2N table
of bits. For 2048 qubits that's 1 MB instead of 10^616 numbers.

Random Clifford circuits are still excellent scramblers (they mimic random
states closely enough to reproduce the Page curve), so we can simulate
a 2048-qubit toy black hole exactly, on a laptop. We use Stim, a fast
Clifford simulator from Google.

The catch: Clifford gates alone can't do everything a quantum computer can,
and their entropies are always whole numbers of bits. So this is a slightly
idealised black hole, but a genuinely simulated one.
"""

import numpy as np
import stim

# There are only 11,520 distinct two-qubit Clifford gates, and drawing one is
# slower than applying it, so we draw a pool once and pick from it at random.
_GATE_POOL_SIZE = 20000


class CliffordBlackHole:
    def __init__(self, n_qubits, seed=0):
        self.n = n_qubits
        self.depth = 0
        self.rng = np.random.default_rng(seed)
        self.sim = stim.TableauSimulator(seed=seed)
        self.sim.set_num_qubits(n_qubits)
        self.pool = [stim.Tableau.random(2) for _ in range(_GATE_POOL_SIZE)]

    def apply_layer(self):
        """
        One layer: pair the qubits up at random (any qubit can meet any other,
        as black holes are thought to allow) and apply a random two-qubit
        Clifford gate to each pair.
        """
        n = self.n
        perm = self.rng.permutation(n)
        pairs = [(int(perm[i]), int(perm[i + 1])) for i in range(0, n - 1, 2)]
        choices = self.rng.integers(len(self.pool), size=len(pairs))
        for (a, b), g in zip(pairs, choices):
            self.sim.do_tableau(self.pool[g], [a, b])
        self.depth += 1

    def stabilizers(self):
        """The state's N stabilizers as two N x N bit arrays: X part and Z part."""
        tableau = self.sim.current_inverse_tableau().inverse()
        _, _, z2x, z2z, _, _ = tableau.to_numpy()
        return z2x, z2z

    def page_curve(self):
        return prefix_entropies(*self.stabilizers())


def prefix_entropies(x_bits, z_bits):
    """
    Entanglement entropy (bits) of radiation = qubits 0..k-1, for every k at once.

    For a stabilizer state:  S(A) = rank(stabilizers cut down to A) - |A|
    where rank is over bits (arithmetic mod 2). Order the columns as
    x0 z0 x1 z1 ... and run Gaussian elimination column by column, left to
    right: the number of pivots found in the first 2k columns is exactly the
    rank of the stabilizers restricted to qubits 0..k-1. So a single
    elimination gives the whole Page curve.
    """
    n = x_bits.shape[0]
    matrix = np.empty((n, 2 * n), dtype=bool)
    matrix[:, 0::2] = x_bits
    matrix[:, 1::2] = z_bits

    # Pack each row's bits into bytes (padded to whole 64-bit words) so that
    # adding two rows mod 2 is a fast XOR of a few machine words.
    packed = np.packbits(matrix, axis=1, bitorder="little")
    pad = (-packed.shape[1]) % 8
    packed = np.ascontiguousarray(np.pad(packed, ((0, 0), (0, pad))))
    words = packed.view(np.uint64)

    unused = np.ones(n, dtype=bool)  # rows not yet chosen as a pivot
    rank_upto = np.zeros(2 * n + 1, dtype=np.int64)
    rank = 0
    for col in range(2 * n):
        has_bit = (packed[:, col >> 3] & (1 << (col & 7))).astype(bool) & unused
        rows = np.flatnonzero(has_bit)
        if rows.size:
            pivot = rows[0]
            unused[pivot] = False
            words[rows[1:]] ^= words[pivot]  # clear this column from the other rows
            rank += 1
        rank_upto[col + 1] = rank

    k = np.arange(n + 1)
    return rank_upto[2 * k] - k
