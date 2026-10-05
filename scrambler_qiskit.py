"""The scrambling circuit built with IBM's Qiskit."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def build_circuit(scrambler):
    """Turn a Scrambler recipe into a Qiskit QuantumCircuit."""
    qc = QuantumCircuit(scrambler.n_qubits)
    for layer_angles, layer_pairs in zip(scrambler.angles, scrambler.pairs):
        for qubit, (a, b, c) in enumerate(layer_angles):
            qc.rz(a, qubit)
            qc.ry(b, qubit)
            qc.rz(c, qubit)
        for q1, q2 in layer_pairs:
            qc.cz(q1, q2)
        qc.barrier()  # only affects drawing: keeps each layer visually separate
    return qc


def final_state(scrambler):
    """Simulate the circuit exactly and return the 2^n amplitudes."""
    return np.asarray(Statevector(build_circuit(scrambler)).data)
