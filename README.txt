README
Fast calculation of high-dimensional Siegel theta functions (g>20) for structured period matrices (soliton solutions and more).

Precision and Accuracy
Unlike heuristic approximations, this method is mathematically exact.
The factorization is based on a rigorous identity for theta functions,
meaning it introduces zero theoretical error when the period matrix is perfectly block-diagonal.
The only deviations are those inherent to standard floating-point arithmetic.

S(2,2)-Based Acceleration of High-Dimensional Riemann Theta Functions
Benchmark Script for naive and s22 modes.
This repository provides a benchmark implementation for computing high-dimensional Riemann theta functions using two modes:

s22   : S(2,2) decomposition (two naive theta evaluations of sizes g1 and g2)
naive : full naive theta evaluation in dimension g

The goal is to demonstrate the dramatic computational speedup obtained by exploiting the block structure that appears on the S(2,2) locus of the GL(4) Hitchin system.

For example, for g = 17 and N_cut = 2:

naive (17D) estimated runtime: about 3,900,000 seconds (about 45 days)
s22 (8D + 9D) measured runtime: about 13 seconds
This corresponds to an effective speedup of roughly 1e5 times.

--
1.Features

Two computation modes:

s22: split g into (g1, g2) and compute two naive theta functions
naive: full naive theta computation in dimension g

Automatic resume:
results are stored in results.json
completed cases are skipped on the next run

Progress logging:
progress_log.txt records timestamps, progress percentage, ETA, and completion time

Configurable g ranges:
s22_g_max and naive_g_max allow limiting the maximum genus per mode

Environment information stored per result:
platform, processor, Python version, timestamp

Automatic extrapolation:
if naive g=17 is not measured, the script estimates it using scaling ratios

--
2.Background

A naive Riemann theta function in dimension g requires summing
(2 * N_cut + 1)^g
terms.

For g = 17 and N_cut = 2, this becomes:
5^17 = 7.6e11 terms

which is not practical for direct computation.

On the S(2,2) locus, the spectral curve becomes reducible:
Ctilde = Ctilde_1 union Ctilde_2
and the period matrix Omega becomes approximately block diagonal:
Omega approx diag(Omega_1, Omega_2)
This allows the approximation:
theta(z | Omega) approx theta(z1 | Omega_1) * theta(z2 | Omega_2)
reducing the effective dimension from 17 to (g1 + g2) = (8 + 9).

--

3.File Overview

ScanS22.py
Main benchmark script.
Implements s22 and naive modes, resume logic, ETA reporting, and summary output.

config.txt
Configuration file created automatically on first run.

progress_log.txt
Progress log with timestamps and ETA.

results.json
Stores all benchmark results, including:
- mode
- g
- N_cut
- time_s
- n_terms
- value_re, value_im
- environment info

4.Usage

Run the benchmark:
python ScanS22.py

The script will:
- load config.txt
- resume from previous results.json if resume=true
- compute theta functions for all (mode, g, N_cut) combinations
- log progress and ETA
- save results incrementally
- print a summary at the end

--
5.Configuration (config.txt)
Example:

g_list = 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
N_cut_list = 1,2
mode = all
seed = 42
report_every = 1000000
resume = true
s22_g_max = 0
naive_g_max = 0

Meaning:
mode:
s22   : run only S(2,2) split mode
naive : run only naive mode
all   : run s22 first, then naive

s22_g_max, naive_g_max:
If set to a positive integer, skip computations with g > g_max.
If 0 or omitted, no limit.

--
6.Output Summary
At the end of the run, the script prints:
per-mode results sorted by g and N_cut
naive/s22 speed ratios
extrapolated naive g=17 runtime (if not measured)
speedup relative to s22 for g=17

--
7.Notes
The naive g=17 runtime is often too large to measure directly.
In such cases, the script estimates it using the term count ratio:

(2*N_cut + 1)^(17 - g_last_measured)

The s22 runtime for g=17 is always measured directly.

This project was originally motivated by exploring numerical aspects of
the relation between N=2 supergravity attractor equations and the geometry
of GL(4) Hitchin systems.

--

License

MIT License








