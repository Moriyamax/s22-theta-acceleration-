# Quantitative Observation of Cancellation Effects 
# in S(2,2) Theta Function Decomposition

**Masaru Moriyama** (Independent Researcher) — May 2026

Zenodo DOI: [技術ノートのDOIをここに]  
GitHub: [リポジトリURLをここに]

---

## What This Is

An observational report on an anomalous numerical phenomenon 
in the S(2,2) approximation of Riemann theta functions.

When a period matrix Omega lies **outside** S(2,2) 
(i.e., has nonzero off-diagonal entries of magnitude delta),
the S(2,2) factorization becomes an approximation.

The intuitive worst-case error bound is:

    |error| <= delta × b^g × C

which predicts |error| ~ 81 at delta=1.0, g=8, b=3.

**Observed median error: ~10⁻¹⁴**

Discrepancy from worst-case bound: ~18 orders of magnitude.  
Discrepancy from CLT prediction: ~15 orders of magnitude.  
**The CLT hypothesis is quantitatively rejected.**

---

## Key Findings (~30,000 cases, 4 experiments)

| Finding | Result |
|---|---|
| delta-independence | Error median invariant to delta=0.1–1.0 |
| g-dependence | Cancellation grows monotonically with g |
| offblock_norm correlation | Pearson r < 0.07 (effectively zero) |
| Equal split | Optimal in both error and cost |
| Collapse phenomenon | All Omega lock to machine precision at specific (g,delta) |

The cancellation mechanism **exceeds CLT-level suppression 
by 15 orders of magnitude**. Its theoretical origin is unknown.

---

## Why This Matters

This phenomenon is only observable through implementation.
The S(2,2) factorization identity is mathematically trivial
(proof follows directly from the definition of theta functions),
which is precisely why no systematic implementation existed —
and why this anomalous cancellation has gone unnoticed.

**To our knowledge, this is the first published dataset 
quantifying this effect.**

---

## Open Problems

- Why does cancellation exceed CLT prediction by 15 orders?
- What causes the 'collapse' phenomenon at specific (g, delta)?
- Can this be formulated as a probabilistic error bound?
- Is this related to exponential sum cancellation 
  (Weyl, van der Corput, Deligne/Weil)?

---

## Contents

| File | Description |
|---|---|
| `cancellation_report_en.pdf` | This technical note |
| `cancellation_scan.py` | Experiments 1 & 2 (delta scan, offblock_norm) |
| `asymsplit_scan.py` | Experiment 3 (asymmetric split) |
| `omega_fixed_scan.py` | Experiment 4 (variance decomposition) |
| `skk_results.json` | Raw experimental data |

---

## Reproduce

```bash
python cancellation_scan.py    # ~30 min, 12 threads
python asymsplit_scan.py
python omega_fixed_scan.py
```

Environment: Python 3.13, NumPy, SciPy, Matplotlib  
Hardware: Ryzen 5 (12 threads), 16GB RAM

---

## Keywords

Riemann theta function, S(2,2) decomposition, 
block-diagonal period matrix, cancellation effect,
exponential sum cancellation, anomalous error suppression,
beyond CLT, numerical benchmark, Siegel upper half-space,
lattice sum, complex phase cancellation

---

## Related

- [Technical Note: S(k,k) Factorization Infrastructure] 
  (Zenodo DOI: xxxx) — main paper
- [S(k,k) Series Benchmark] 
  (Zenodo DOI: xxxx) — S(3,3), S(5,5) comparison

## Citation

Moriyama, M. (2026). Quantitative Observation of 
Cancellation Effects in S(2,2) Theta Function Decomposition. 
Zenodo. DOI: xxxx
