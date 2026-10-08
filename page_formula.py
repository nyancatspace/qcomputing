"""
Page's formula, on its own.

A black hole made of N qubits has emitted k of them as radiation.
How entangled is that radiation with what's left of the black hole?
Don Page's 1993 answer, for a black hole that scrambles information perfectly:

    S = [ 1/(n+1) + 1/(n+2) + ... + 1/(m*n) ]  -  (m - 1) / (2n)

where m = 2^(the smaller part) and n = 2^(the bigger part), so that m <= n.
(S comes out in "nats"; dividing by ln 2 converts to bits.)

The sum can have an astronomically large number of terms, so we use a
shortcut from calculus, the digamma function:
    1/(n+1) + ... + 1/(m*n) = digamma(m*n + 1) - digamma(n + 1)

Usage:
    .venv/bin/python page_formula.py 2048          # table for a 2048-qubit black hole
    .venv/bin/python page_formula.py 2048 1024     # just one value: k = 1024
"""

import sys

import mpmath

mpmath.mp.dps = 50  # work with 50 significant digits


def page_entropy(k, N):
    """Entropy (in bits) of the radiation after k of N qubits have been emitted."""
    small, big = sorted((k, N - k))
    m = mpmath.mpf(2) ** small
    n = mpmath.mpf(2) ** big
    nats = mpmath.digamma(m * n + 1) - mpmath.digamma(n + 1) - (m - 1) / (2 * n)
    return nats / mpmath.log(2)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

    if len(sys.argv) > 2:
        k = int(sys.argv[2])
        print(f"N = {N} qubits, k = {k} emitted:  S = {mpmath.nstr(page_entropy(k, N), 10)} bits")
        return

    # A handful of points along the evaporation, including around the halfway point.
    half = N // 2
    ks = sorted({0, 1, N // 4, half - 3, half - 1, half, half + 1, half + 3, 3 * N // 4, N - 1, N})
    print(f"Page curve for a black hole of N = {N} qubits\n")
    print(f"{'emitted k':>10} {'entropy S (bits)':>18} {'most possible':>14}")
    for k in ks:
        if 0 <= k <= N:
            print(f"{k:>10} {mpmath.nstr(page_entropy(k, N), 10):>18} {min(k, N - k):>14}")


if __name__ == "__main__":
    main()
