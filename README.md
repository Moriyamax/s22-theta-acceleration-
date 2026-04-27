# RLD_theta_engine.jl and RLD_theta_engine.py

A high-performance Julia implementation for evaluating Riemann Theta Functions in ultra-high dimensions (genus g > 20,000) using the Recursive Log-Decomposition (RLD) algorithm.

## Overview

The Riemann theta function is central to mathematical physics and algebraic geometry, but its numerical evaluation typically suffers from the "curse of dimensionality," with complexity growing exponentially: O(b^g).

This engine breaks the exponential barrier by leveraging addition formulas for period matrices with a block-diagonal structure (S(2,2) locus). By recursively decomposing the genus into smaller subspaces and performing calculations in the complex log-domain, it achieves exact results for high-genus regimes that were previously computationally inaccessible.

## Key Features

- Recursive Log-Decomposition: Decomposes a genus-g problem into smaller dimensions, reducing complexity from exponential to near-linear for specific matrix structures.
- Log-Domain Arithmetic: All internal calculations are performed as log(theta) to prevent numerical overflow, which is inevitable in high dimensions (e.g., g=1,500 often yields results exceeding 10^300).
- Optimal Dimensioning (Padding): Automatically pads any genus g to the nearest 2^n dimension to optimize the recursive binary tree path.
- High-Parallelism: Utilizes Julia's multi-threading (Threads.@spawn) to evaluate sub-dimensions concurrently.

## Performance Benchmarks

Based on execution logs on a standard multi-core environment:

| Genus (g) | Evaluation Time | Result Type |
| :--- | :--- | :--- |
| 16 | ~0.17s | Standard Float64 |
| 1,500 | ~0.13s | Log-Domain only |
| 20,000 | ~11.58s | Log-Domain only |

Note: For g=20,000, the absolute value of the theta function is so large that it can only be represented in the log-domain (e.g., Re(log theta) approx 10,000).

## Usage

### Prerequisites
- Julia 1.9+
- Standard libraries: LinearAlgebra, Random, Printf, JSON.
  
  or

- Python 3.13.12
- numpy

### Running the Engine

julia --threads auto RLD_theta_engine.jl

## Technical Details
Recursive Formula
The core logic utilizes the property that for block-diagonal Omega, the theta function factorizes:

log theta(z, Omega) = log theta(z_1, Omega_1) + log theta(z_2, Omega_2)

This engine implements this via log_theta_recursive, which handles the cross-terms and recursive steps efficiently.

## Configuration
You can adjust the test parameters in the main() function:

g_list: List of dimensions to test (e.g., [16, 20000]).
N_cut: Summation limit for the base-case naive calculation.
use_log_output: Set to true to display results in log-format, avoiding conversion errors for huge numbers.

## Repository
Developed as part of the s22-theta-acceleration project.
