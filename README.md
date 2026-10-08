# Toy black holes on a (simulated) quantum computer

Beginner-friendly experiments on the black hole information paradox, using
small numbers of qubits as a stand-in for a black hole.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Stage 1: the Page curve (`page_curve.py`)

Models a perfectly scrambled black hole as a random 12-qubit state and lets it
"evaporate" one qubit at a time. The entanglement entropy of the radiation
rises, peaks halfway (the Page time), and falls back to zero, which is the
signature of information escaping rather than being destroyed.

```
.venv/bin/python page_curve.py      # -> page_curve.png
```

## Stage 2: how fast does it scramble? (`stage2_scrambling.py`)

Builds the scrambling out of real quantum gates: layers of random single-qubit
rotations plus CZ gates. The same circuit is written in both **Qiskit**
(`scrambler_qiskit.py`) and **NVIDIA CUDA-Q** (`scrambler_cudaq.py`), and the
script checks that the two produce identical quantum states.

It compares two ways of connecting the qubits:

- **chain**: only neighbours interact. Scrambling is slow (12 qubits still
  aren't fully scrambled after 40 layers).
- **all-to-all**: any pair can interact. Fully scrambled after about 8 layers.

The "fast scrambling" conjecture says black holes behave like the all-to-all case.

```
.venv/bin/python stage2_scrambling.py                  # CPU, -> scrambling.png
.venv/bin/python stage2_scrambling.py --target nvidia  # CUDA-Q on an NVIDIA GPU
```

## Stage 2b: a 2048-qubit black hole (`big_black_hole.py`)

A full state vector for 2048 qubits would need 2^2048 (about 10^616)
numbers. Using only Clifford gates, the state fits in a 2048 x 4096 table of
bits (the Gottesman–Knill theorem), so we simulate it exactly with
[Stim](https://github.com/quantumlib/Stim) (`clifford_scrambler.py`). The script:

- evaluates **Page's exact formula** at 2048 qubits, and for a Sun-mass black
  hole (~10^77 bits). At every size the curve is the triangle min(k, N−k),
  short by at most 1/(2 ln 2) ≈ 0.72 bits at the Page time;
- simulates a 2048-qubit black hole and lands on that formula;
- measures how many layers it takes to scramble as N grows from 16 to 2048:
  about **1.5·N** layers for the chain (3168 at N = 2048) but only about
  **log₂ N** for all-to-all (11 at N = 2048), which is the fast-scrambling behaviour.

```
.venv/bin/python big_black_hole.py              # ~5 min, -> big_black_hole.png
.venv/bin/python big_black_hole.py --plot-only  # redraw from saved results
```

## Next

Stage 3: the Hayden–Preskill protocol. Throw a qubit into the black hole and
recover it from the radiation.
