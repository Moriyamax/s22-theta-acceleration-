---
title: 's22-theta-acceleration: A Python/Julia library for block-diagonal acceleration of high-dimensional Siegel theta function computation'
tags:
  - Python
  - Julia
  - Siegel theta functions
  - Riemann theta functions
  - numerical computation
  - block-diagonal decomposition
  - algebraic geometry
authors:
  - name: Masaru Moriyama
    affiliation: '1'
affiliations:
  - index: 1
    name: Independent Researcher, Japan
date: 25 April 2026
bibliography: paper.bib
---

# Summary

Computing Siegel theta functions in high dimensions is a fundamental bottleneck in
several areas of mathematical physics and algebraic geometry. The naive truncated
series requires summing $(2N_\mathrm{cut}+1)^g$ terms, where $g$ is the dimension,
causing exponential growth in computation time. On commodity hardware, a naive
evaluation at $g=13$ with $N_\mathrm{cut}=2$ already takes over two hours.

`s22-theta-acceleration` implements a **block-diagonal (S(2,2)) decomposition**
that exploits an algebraic factorization property: when the period matrix $\Omega$
is block-diagonal, the Siegel theta function factors exactly as a product of
lower-dimensional theta functions. The library achieves speedups of up to **41,025×**
at $g=14$ compared to naive computation, and scales to $g=25$ on standard desktop
hardware. Both Python and Julia implementations are provided, confirming that the
speedup is language-independent and stems from the mathematical structure of the
decomposition.

# Statement of Need

Siegel theta functions $\theta(z|\Omega)$ appear in the construction of
quasi-periodic solutions to integrable PDEs (e.g., KdV, NLS equations), in the
study of abelian varieties, and in the computation of period matrices of algebraic
curves [@Dubrovin1981; @Deconinck2004]. For many practical applications,
repeated evaluation at many values of $z$ for a fixed $\Omega$ is required.

The computational cost of naive evaluation grows as $\mathcal{O}(b^g)$ where
$b = 2N_\mathrm{cut}+1$. For $g \geq 13$ and $N_\mathrm{cut}=2$, this becomes
intractable on standard hardware. Existing software—including the Maple `algcurves`
package [@Deconinck2004], the Sage/Python implementation [@Swierczewski2016], and
the FLINT `acb_theta` module [@Kieffer2025]—handles the general case but does not
specifically optimize for block-diagonal period matrices.

For applications where $\Omega$ has (or is well approximated by) a block-diagonal
structure—such as reducible spectral curves in Hitchin systems, or period matrices
near the boundary of the Siegel moduli space—the present library provides a
targeted, lightweight alternative that achieves machine-precision accuracy on the
block-diagonal locus while running orders of magnitude faster.

The target audience is researchers in:

- **Integrable systems and nonlinear Fourier analysis**: rapid evaluation of
  theta functions for fiber-optic signal processing [@Chimmalgi2023].
- **Algebraic geometry**: computation involving reducible curves and their
  Jacobians.
- **Numerical experimentation**: fast prototyping before committing to
  arbitrary-precision libraries.

# State of the Field

| Software | Approach | Precision | Max practical $g$ | Block-diagonal opt. |
|---|---|---|---|---|
| Deconinck et al. / Maple [@Deconinck2004] | Ellipsoid truncation | Double / arbitrary | ~10 (double) | No |
| Swierczewski–Deconinck / Sage [@Swierczewski2016] | Same | Double | ~10 | No |
| FLINT `acb_theta` [@Kieffer2025] | Quasi-linear AGM | Arbitrary | High (arbitrary prec.) | Implicit via dim reduction |
| Chimmalgi–Wahls HC [@Chimmalgi2023] | Hyperbolic cross | Double (~1%) | 60 (moderate accuracy) | No |
| **This work** | Block-diagonal factorization | Double (machine precision on locus) | **25** (double, $N_\mathrm{cut}=2$) | **Yes (explicit)** |

The key distinction from all prior work is that `s22-theta-acceleration` makes the
block-diagonal factorization **explicit and user-controllable**, and provides
benchmarks up to $g=25$ demonstrating the resulting speedup. FLINT's `acb_theta`
implicitly handles eigenvalue imbalance via dimension reduction [@FLINTdoc], but
does not expose block-diagonal factorization as a standalone, lightweight interface
for double-precision use.

Compared to Chimmalgi–Wahls [@Chimmalgi2023], the present approach achieves
machine-precision accuracy (relative error $\sim 10^{-16}$) on the block-diagonal
locus, versus their ~1% accuracy at $g=60$. The two methods are thus complementary:
this library is suited for high-accuracy evaluation near the block-diagonal locus,
while the hyperbolic-cross method targets moderate-accuracy evaluation at very high
genus.

# Software Design

## Core Algorithm

When $\Omega = \mathrm{diag}(\Omega_1, \Omega_2)$ (off-diagonal blocks exactly
zero), the Siegel theta function factorizes algebraically:

$$\theta(z|\Omega) = \theta_1(z_1|\Omega_1) \times \theta_2(z_2|\Omega_2).$$

This is an exact identity, not an approximation. The S(2,2) decomposition
implements this factorization, splitting a $g$-dimensional sum into two
$\lfloor g/2 \rfloor$- and $\lceil g/2 \rceil$-dimensional sums. The reduction in
floating-point operations is $\Theta(b^{g/2})$, verified both theoretically
(Proposition 2.1 in the companion preprint) and empirically across $g=2$–$14$.

The equal split $g_1 = \lfloor g/2 \rfloor$, $g_2 = \lceil g/2 \rceil$ is
cost-optimal for fixed $N_\mathrm{cut}$, as it minimizes $b^{g_1} + b^{g_2}$
subject to $g_1 + g_2 = g$.

## Accuracy on the Block-Diagonal Locus

On the S(2,2) locus (off-diagonal blocks exactly zero), the relative error between
naive and S(2,2) evaluations is at the level of floating-point machine epsilon
($\approx 2.22 \times 10^{-16}$) for all tested cases ($g=2$–$14$,
$N_\mathrm{cut}=1,2$). This confirms that no numerical degradation is introduced
by the factorization.

## Approximation Off the Locus

For general $\Omega$ (off-diagonal blocks nonzero), the factorization becomes an
approximation. The error grows monotonically with the magnitude of off-diagonal
blocks, as demonstrated numerically in Appendix I of the companion preprint. The
truncation error scales as $\log_{10}(\varepsilon) \approx -({\pi}/{\ln 10})
\times \lambda_\mathrm{min}(\mathrm{Im}\,\Omega)$ for fixed $N_\mathrm{cut}=1$,
consistent with the classical convergence theory [@Frauendiener2017; @Deconinck2004].

## Implementation

The library provides two independent implementations:

- **Python** (`scanS22.py`, `same_omega_comparison.py`): uses NumPy, runs on
  any platform. Benchmarked on Windows (AMD Ryzen 5 5500GT, Ryzen 7 5700X,
  Intel N97).
- **Julia** (`SingleThetaBench.jl`): ~100× faster than Python for naive
  computation; S(2,2) speedup ratios are consistent with Python (Table D2 of
  companion preprint), confirming language independence of the algorithmic gain.

Additional features include: resume/checkpoint support for long runs,
configuration via `config.txt`, and JSON output with full environment metadata.

# Research Impact Statement

The software is accompanied by a preprint [@Moriyama2026] providing full
benchmarks. Key reproducible results:

- **g=14, N_cut=2**: 41,025× speedup (naive: 39,160 s vs. S(2,2): 0.955 s),
  AMD Ryzen 5 5500GT, single-threaded Python.
- **g=25, N_cut=2**: 12,665 s with S(2,2) on AMD Ryzen 7 5700X (naive
  intractable).
- All benchmark scripts and raw results are archived in the GitHub repository.

The library fills a gap for researchers who need fast, double-precision theta
function evaluation on block-diagonal period matrices without the overhead of
arbitrary-precision libraries such as FLINT. Anticipated use cases include
parameter scanning in integrable PDE applications and numerical experimentation
in algebraic geometry.

Planned extensions include: parallel computation (the two sub-problems are
independent), Julia implementation at $g=60$ for direct comparison with
[@Chimmalgi2023], and generalization to $k$-block decompositions
$S_{(n_1,\ldots,n_k)}$.

# AI Usage Disclosure

Generative AI tools (Claude, Anthropic) were used to assist with manuscript
drafting and review simulation. All mathematical claims, numerical results, and
code were produced and verified by the author independently. No AI-generated code
is present in the software repository.

# Acknowledgements

The author thanks the developers of NumPy, Julia, and the open-source mathematics
community whose tools made this work possible. No external funding was received.

# References
