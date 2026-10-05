"""
The scrambling circuit built with NVIDIA's CUDA-Q.

CUDA-Q compiles the decorated kernel function below into a quantum program.
It runs on a CPU simulator by default, or on NVIDIA GPUs with
set_target("nvidia"), which is what makes simulating 30+ qubits practical.
The kernel code doesn't change between the two.
"""

import warnings

import numpy as np

with warnings.catch_warnings():
    # CUDA-Q prints preview/deprecation notices on import that aren't relevant here.
    warnings.simplefilter("ignore")
    import cudaq


@cudaq.kernel
def scrambler_kernel(n_qubits: int, depth: int, angles: list[float],
                     pair_qubits: list[int], layer_starts: list[int]):
    q = cudaq.qvector(n_qubits)
    for layer in range(depth):
        for i in range(n_qubits):
            base = 3 * (layer * n_qubits + i)
            rz(angles[base], q[i])
            ry(angles[base + 1], q[i])
            rz(angles[base + 2], q[i])
        for p in range(layer_starts[layer], layer_starts[layer + 1]):
            cz(q[pair_qubits[2 * p]], q[pair_qubits[2 * p + 1]])


def set_target(name):
    """Pick a CUDA-Q simulator, e.g. "qpp-cpu" (CPU) or "nvidia" (GPU)."""
    cudaq.set_target(name)


def target_name():
    return cudaq.get_target().name


def final_state(scrambler):
    """Simulate the circuit exactly and return the 2^n amplitudes."""
    angles, pair_qubits, layer_starts = scrambler.flat_args()
    state = cudaq.get_state(scrambler_kernel, scrambler.n_qubits, scrambler.depth,
                            angles, pair_qubits, layer_starts)
    return np.array(state)
