"""
cancellation_scan.py  ― 打ち消し効果定量化パラメータスキャン

S_{(2,2)} 近似誤差の δ 依存性を g・delta を振りながら系統的に測定する。

設定:
  G_LIST   = [8, 10, 12, 14]
  N_CUT    = 1
  DELTAS   = 0.1, 0.2, ..., 1.0  (10点)
  N_SEED   = 100  (各 (g, delta) 組み合わせごとに 100 個のランダム Ω)

実装ベース:
  appendix_i_plots.py の make_omega / theta_naive をそのまま継承。
  動作確認済みのロジックを流用し、numpy vectorized 化と multiprocessing
  による並列化を加えた。

定義式 (Deconinck et al. 2004 / appendix_i_plots.py と完全一致):
  theta(z|Omega) = sum_{n: |n_i|<=N_cut}
                      exp(pi*i * n^T Omega n + 2*pi*i * n^T z)

周期行列生成 (appendix_i_plots.py と完全一致):
  Im_block = A @ A.T + g_block * I   (正定値保証)
  Re_block = symmetric random
  Omega_block = Re_block + i * Im_block
  off-diagonal (delta > 0):
    C = delta * (randn + i*randn)  (複素摂動)

出力:
  cancellation_results.json  ― 全数値データ + サマリー統計
  cancellation_scan.log      ― 進捗ログ

実行方法:
  python cancellation_scan.py
  python cancellation_scan.py --workers 6   # コア数指定
  python cancellation_scan.py --resume      # 中断再開
"""

import numpy as np
import json
import time
import os
import sys
import argparse
import logging
import multiprocessing
from itertools import product as iproduct
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
G_LIST      = [8, 10, 12, 14]
N_CUT       = 1
DELTAS      = [round(0.1 * k, 1) for k in range(1, 11)]   # 0.1〜1.0
N_SEED      = 100
SEED_BASE   = 2000   # appendix_i_plots.py (SEED_OFFSET=100) と区別

RESULT_PATH = "cancellation_results.json"
LOG_PATH    = "cancellation_scan.log"
CHUNK_SIZE  = 200_000   # theta_naive の vectorized chunk サイズ

# ─────────────────────────────────────────
# ログ設定
# ─────────────────────────────────────────
def setup_logging():
    fmt = "[%(asctime)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 周期行列生成 (appendix_i_plots.py と同一ロジック)
# ─────────────────────────────────────────
def make_omega(g: int, delta: float, seed: int):
    """
    Siegel 上半空間の周期行列を生成する。
    appendix_i_plots.py の make_omega と完全に同一のロジック。
    Im(Omega) の正定値性を assert で保証。
    """
    rng = np.random.default_rng(seed)
    g1, g2 = g // 2, g - g // 2

    A   = rng.standard_normal((g1, g1))
    Im1 = A @ A.T + g1 * np.eye(g1)
    Re1 = rng.standard_normal((g1, g1)); Re1 = (Re1 + Re1.T) / 2
    O1  = Re1 + 1j * Im1

    B   = rng.standard_normal((g2, g2))
    Im2 = B @ B.T + g2 * np.eye(g2)
    Re2 = rng.standard_normal((g2, g2)); Re2 = (Re2 + Re2.T) / 2
    O2  = Re2 + 1j * Im2

    Omega = np.zeros((g, g), dtype=complex)
    Omega[:g1, :g1] = O1
    Omega[g1:, g1:] = O2

    if delta > 0.0:
        C = delta * (rng.standard_normal((g1, g2))
                     + 1j * rng.standard_normal((g1, g2)))
        Omega[:g1, g1:] = C
        Omega[g1:, :g1] = C.conj().T

    z = rng.standard_normal(g)

    # 正定値確認
    eigs = np.linalg.eigvalsh(Omega.imag)
    assert np.all(eigs > 0), \
        f"Im(Omega) not positive definite! min_eig={eigs.min():.4e} " \
        f"(g={g}, delta={delta}, seed={seed})"

    return Omega, z

# ─────────────────────────────────────────
# theta 関数 (numpy vectorized + chunk 分割)
# ─────────────────────────────────────────
def theta_naive(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    """
    定義式: sum_{n: |n_i|<=N_cut} exp(pi*i*n^T Omega n + 2*pi*i*n^T z)
    appendix_i_plots.py の theta_naive と等価。
    numpy の einsum で chunk ごとに vectorize して高速化。
    """
    g     = len(z)
    b     = 2 * N_cut + 1
    total = b ** g
    result = 0j

    # 格子点イテレータを chunk ごとに消費
    it  = iproduct(range(-N_cut, N_cut + 1), repeat=g)
    buf = []
    for n in it:
        buf.append(n)
        if len(buf) >= CHUNK_SIZE:
            arr = np.array(buf, dtype=np.float64)
            qc  = np.einsum('ij,ij->i', arr @ Omega, arr)
            lc  = arr @ z
            result += np.sum(np.exp(np.pi * 1j * qc + 2 * np.pi * 1j * lc))
            buf = []
    if buf:
        arr = np.array(buf, dtype=np.float64)
        qc  = np.einsum('ij,ij->i', arr @ Omega, arr)
        lc  = arr @ z
        result += np.sum(np.exp(np.pi * 1j * qc + 2 * np.pi * 1j * lc))

    return result


def theta_s22(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    """
    S_{(2,2)} 因子化: theta(z|Omega) = theta(z1|Omega1) * theta(z2|Omega2)
    Omega が block-diagonal のときは厳密（代数的恒等式）。
    delta > 0 のときは近似。
    """
    g1 = len(z) // 2
    v1 = theta_naive(z[:g1],     Omega[:g1, :g1],         N_cut)
    v2 = theta_naive(z[g1:],     Omega[g1:, g1:],         N_cut)
    return v1 * v2

# ─────────────────────────────────────────
# 1ケース実行（multiprocessing の worker 関数）
# ─────────────────────────────────────────
def run_case(args):
    """
    引数: (g, delta, delta_idx, seed_i)
    戻り値: dict（1ケース分の数値データ）
    """
    g, delta, delta_idx, seed_i = args
    seed = SEED_BASE + g * 10000 + delta_idx * 100 + seed_i

    try:
        Omega, z = make_omega(g, delta, seed)
    except AssertionError as e:
        return {
            "g": g, "delta": delta, "seed": seed, "seed_i": seed_i,
            "error": str(e),
            "rel_err": float("nan"), "log10_rel_err": float("nan"),
            "im_diag_mean": float("nan"), "offblock_norm": float("nan"),
        }

    # 補助統計
    g1 = g // 2
    im_diag_mean  = float(np.mean(np.diag(Omega.imag)))
    offblock_norm = float(
        np.linalg.norm(Omega.imag[:g1, g1:]) / np.sqrt(g1 * (g - g1))
    ) if delta > 0.0 else 0.0

    vn = theta_naive(z, Omega, N_CUT)
    vs = theta_s22(z,  Omega, N_CUT)

    rel_err  = float(abs(vn - vs) / abs(vn)) if abs(vn) > 1e-300 else float("nan")
    log_rel  = float(np.log10(rel_err)) if (not np.isnan(rel_err) and rel_err > 0) \
               else -16.0

    return {
        "g":             g,
        "delta":         delta,
        "seed":          seed,
        "seed_i":        seed_i,
        "rel_err":       rel_err,
        "log10_rel_err": log_rel,
        "im_diag_mean":  im_diag_mean,
        "offblock_norm": offblock_norm,
        "val_naive_re":  float(vn.real),
        "val_naive_im":  float(vn.imag),
        "val_s22_re":    float(vs.real),
        "val_s22_im":    float(vs.imag),
    }

# ─────────────────────────────────────────
# サマリー統計計算
# ─────────────────────────────────────────
def compute_summary(records):
    summary = {}
    for g in G_LIST:
        for delta in DELTAS:
            key    = f"g{g}_d{delta:.1f}"
            subset = [r for r in records
                      if r["g"] == g and abs(r["delta"] - delta) < 1e-9]
            errs   = [r["log10_rel_err"] for r in subset
                      if not np.isnan(r["log10_rel_err"])]
            rerrs  = [r["rel_err"] for r in subset
                      if not np.isnan(r["rel_err"])]
            summary[key] = {
                "g":             g,
                "delta":         delta,
                "n":             len(subset),
                "n_valid":       len(errs),
                "median_log10":  float(np.median(errs))  if errs  else float("nan"),
                "mean_log10":    float(np.mean(errs))    if errs  else float("nan"),
                "std_log10":     float(np.std(errs))     if errs  else float("nan"),
                "min_log10":     float(np.min(errs))     if errs  else float("nan"),
                "max_log10":     float(np.max(errs))     if errs  else float("nan"),
                "q25_log10":     float(np.percentile(errs, 25)) if errs else float("nan"),
                "q75_log10":     float(np.percentile(errs, 75)) if errs else float("nan"),
                "median_relerr": float(np.median(rerrs)) if rerrs else float("nan"),
            }
    return summary

# ─────────────────────────────────────────
# 結果保存
# ─────────────────────────────────────────
def save_results(records, summary, meta):
    out = {"meta": meta, "summary": summary, "records": records}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

def load_results():
    if not Path(RESULT_PATH).exists():
        return []
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])

# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="cancellation_scan.py")
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() or 1),
                        help="並列ワーカー数 (default: CPU コア数)")
    parser.add_argument("--resume", action="store_true",
                        help="中断した計算を再開する")
    args = parser.parse_args()

    setup_logging()
    t_start = time.perf_counter()

    log.info("=" * 65)
    log.info("cancellation_scan.py  打ち消し効果定量化スキャン")
    log.info(f"Python {sys.version.split()[0]}  workers={args.workers}")
    log.info(f"g={G_LIST}  N_cut={N_CUT}  delta={DELTAS}")
    log.info(f"N_seed={N_SEED}  総ケース数="
             f"{len(G_LIST) * len(DELTAS) * N_SEED}")
    log.info(f"resume={args.resume}")
    log.info("=" * 65)

    # 全タスクリスト生成
    all_tasks = [
        (g, delta, delta_idx, seed_i)
        for g              in G_LIST
        for delta_idx, delta in enumerate(DELTAS)
        for seed_i         in range(N_SEED)
    ]
    total = len(all_tasks)

    # resume: 完了済みキーをセットとして保持
    done_records = []
    done_keys    = set()
    if args.resume:
        done_records = load_results()
        for r in done_records:
            done_keys.add((r["g"], round(r["delta"], 1), r["seed_i"]))
        log.info(f"resume: {len(done_keys)} 件スキップ")

    pending = [t for t in all_tasks
               if (t[0], round(t[1], 1), t[3]) not in done_keys]
    log.info(f"実行対象: {len(pending)} / {total} ケース")

    results = list(done_records)
    completed = len(done_records)

    # multiprocessing で並列実行
    # chunksize を大きめにしてオーバーヘッド削減
    chunksize = max(1, len(pending) // (args.workers * 4))

    with multiprocessing.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(run_case, pending, chunksize=chunksize):
            results.append(r)
            completed += 1

            if "error" in r:
                log.warning(f"[{completed}/{total}] ERROR g={r['g']} "
                            f"delta={r['delta']} seed_i={r['seed_i']}: {r['error']}")
            elif completed % 100 == 0 or completed == total:
                pct = 100.0 * completed / total
                log.info(f"[{completed}/{total}] ({pct:.1f}%) "
                         f"g={r['g']} δ={r['delta']:.1f} "
                         f"seed_i={r['seed_i']:3d} "
                         f"log10_err={r['log10_rel_err']:.2f}")

            # 500件ごとに中間保存
            if completed % 500 == 0:
                summary = compute_summary(results)
                meta = {
                    "script":      "cancellation_scan.py",
                    "status":      "running",
                    "g_list":      G_LIST,
                    "N_cut":       N_CUT,
                    "deltas":      DELTAS,
                    "N_seed":      N_SEED,
                    "seed_base":   SEED_BASE,
                    "total_cases": total,
                    "completed":   completed,
                    "workers":     args.workers,
                    "elapsed_s":   round(time.perf_counter() - t_start, 1),
                    "timestamp":   datetime.now().isoformat(timespec="seconds"),
                }
                save_results(results, summary, meta)
                log.info(f"  → 中間保存: {completed} 件完了")

    elapsed = time.perf_counter() - t_start

    # 最終サマリー
    summary = compute_summary(results)
    meta = {
        "script":      "cancellation_scan.py",
        "status":      "completed",
        "g_list":      G_LIST,
        "N_cut":       N_CUT,
        "deltas":      DELTAS,
        "N_seed":      N_SEED,
        "seed_base":   SEED_BASE,
        "total_cases": total,
        "completed":   completed,
        "workers":     args.workers,
        "elapsed_s":   round(elapsed, 1),
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
    }
    save_results(results, summary, meta)

    # コンソールサマリー表示
    log.info("=" * 65)
    log.info("完了: %d ケース  経過時間 %.1f 分", total, elapsed / 60)
    log.info("出力: %s", RESULT_PATH)
    log.info("=" * 65)
    log.info("")
    log.info(f"{'g':>4} {'delta':>6} {'median':>8} {'mean':>8} "
             f"{'std':>6} {'min':>8} {'max':>8}")
    log.info("-" * 55)
    for g in G_LIST:
        for delta in DELTAS:
            s = summary[f"g{g}_d{delta:.1f}"]
            log.info(
                f"{g:>4} {delta:>6.1f} {s['median_log10']:>8.2f} "
                f"{s['mean_log10']:>8.2f} {s['std_log10']:>6.2f} "
                f"{s['min_log10']:>8.2f} {s['max_log10']:>8.2f}"
            )


if __name__ == "__main__":
    # Windows の multiprocessing に必要
    multiprocessing.freeze_support()
    main()
