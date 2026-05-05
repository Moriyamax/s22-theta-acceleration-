"""
omega_fixed_scan.py  ― Omega 固定・z 変更による誤差依存性分離

同一の周期行列 Omega に対して z を 100 個変えたときの誤差分布を測定し、
「打ち消し効果が Omega の構造に由来するか z に依存するか」を分離する。

設定:
  G_LIST      = [8, 10, 12, 14]
  N_CUT       = 1
  DELTAS      = 0.1, 0.5, 1.0  (代表3点)
  N_OMEGA     = 10   (各 (g, delta) で Omega を 10 個用意)
  N_Z         = 100  (各 Omega で z を 100 個変える)
  総ケース数  = 4 × 3 × 10 × 100 = 12,000

分析の着眼点:
  - 同一 Omega の 100 個の z で誤差分布の std が小さい
      → Omega が誤差を決める (Omega 依存)
  - Omega ごとに分布の中央値が大きく異なる
      → Omega の構造が打ち消し強度を決める (Omega 依存)
  - 同一 Omega 内でも z によって誤差が大きく変わる
      → z 依存の寄与が大きい

定義式・行列生成: cancellation_scan.py と完全一致。
z のみ独立した seed で生成する。

出力:
  omega_fixed_results.json
  omega_fixed_scan.log

実行方法:
  python omega_fixed_scan.py
  python omega_fixed_scan.py --workers 12
  python omega_fixed_scan.py --resume
"""

import numpy as np
import json, time, os, sys, argparse, logging, multiprocessing
from itertools import product as iproduct
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
G_LIST    = [8, 10, 12, 14]
N_CUT     = 1
DELTAS    = [0.1, 0.5, 1.0]   # 代表3点（フルスキャンより軽く、代表性を保つ）
N_OMEGA   = 10                 # Omega の個数
N_Z       = 100                # 各 Omega に対する z の個数
OMEGA_SEED_BASE = 5000         # Omega 生成用 seed
Z_SEED_BASE     = 6000         # z 生成用 seed

RESULT_PATH = "omega_fixed_results.json"
LOG_PATH    = "omega_fixed_scan.log"
CHUNK_SIZE  = 200_000

# ─────────────────────────────────────────
# ログ
# ─────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 周期行列生成 (cancellation_scan.py と同一)
# ─────────────────────────────────────────
def make_omega(g: int, delta: float, seed: int):
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

    eigs = np.linalg.eigvalsh(Omega.imag)
    assert np.all(eigs > 0), \
        f"Im(Omega) not PD: min={eigs.min():.3e} (g={g},delta={delta},seed={seed})"

    return Omega

# ─────────────────────────────────────────
# theta 関数 (cancellation_scan.py と同一)
# ─────────────────────────────────────────
def theta_naive(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    result = 0j
    buf = []
    for n in iproduct(range(-N_cut, N_cut + 1), repeat=len(z)):
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
    g1 = len(z) // 2
    v1 = theta_naive(z[:g1],     Omega[:g1, :g1],         N_cut)
    v2 = theta_naive(z[g1:],     Omega[g1:, g1:],         N_cut)
    return v1 * v2

# ─────────────────────────────────────────
# Worker: 1つの (g, delta, omega_idx) を受け取り
#         N_Z 個の z で誤差を計算して返す
# ─────────────────────────────────────────
def run_omega_case(args):
    """
    1つの Omega に対して N_Z 個の z で誤差を計算する。
    naive は z ごとに計算（Omega は固定）。
    """
    g, delta, delta_idx, omega_idx = args
    omega_seed = OMEGA_SEED_BASE + g * 10000 + delta_idx * 100 + omega_idx

    try:
        Omega = make_omega(g, delta, omega_seed)
    except AssertionError as e:
        return {"g": g, "delta": delta, "omega_idx": omega_idx,
                "omega_seed": omega_seed, "error": str(e), "z_results": []}

    # Omega の特性量
    g1 = g // 2
    im_diag_mean  = float(np.mean(np.diag(Omega.imag)))
    offblock_norm = float(
        np.linalg.norm(Omega.imag[:g1, g1:]) / np.sqrt(g1 * (g - g1))
    ) if delta > 0.0 else 0.0
    # off-diagonal の虚部最大固有値（Omega の「歪み」の指標）
    im_full_eigs  = np.linalg.eigvalsh(Omega.imag)
    im_min_eig    = float(im_full_eigs.min())
    im_max_eig    = float(im_full_eigs.max())

    # N_Z 個の z で誤差計算
    z_results = []
    for z_idx in range(N_Z):
        z_seed = Z_SEED_BASE + g * 10000 + omega_idx * 1000 + z_idx
        rng    = np.random.default_rng(z_seed)
        z      = rng.standard_normal(g)

        vn = theta_naive(z, Omega, N_CUT)
        vs = theta_s22(z,  Omega, N_CUT)

        rel = float(abs(vn - vs) / abs(vn)) if abs(vn) > 1e-300 else float("nan")
        log_rel = float(np.log10(rel)) if (not np.isnan(rel) and rel > 0) else -16.0

        z_results.append({
            "z_idx":         z_idx,
            "z_seed":        z_seed,
            "rel_err":       rel,
            "log10_rel_err": log_rel,
        })

    return {
        "g":             g,
        "delta":         delta,
        "omega_idx":     omega_idx,
        "omega_seed":    omega_seed,
        "im_diag_mean":  im_diag_mean,
        "offblock_norm": offblock_norm,
        "im_min_eig":    im_min_eig,
        "im_max_eig":    im_max_eig,
        "z_results":     z_results,
        # z結果のサマリー（アクセスしやすいよう埋め込む）
        "median_log10":  float(np.median([r["log10_rel_err"] for r in z_results])),
        "mean_log10":    float(np.mean  ([r["log10_rel_err"] for r in z_results])),
        "std_log10":     float(np.std   ([r["log10_rel_err"] for r in z_results])),
        "min_log10":     float(np.min   ([r["log10_rel_err"] for r in z_results])),
        "max_log10":     float(np.max   ([r["log10_rel_err"] for r in z_results])),
    }

# ─────────────────────────────────────────
# サマリー統計
# ─────────────────────────────────────────
def compute_summary(records):
    summary = {}
    for g in G_LIST:
        for delta in DELTAS:
            key    = f"g{g}_d{delta:.1f}"
            subset = [r for r in records
                      if r["g"] == g and abs(r["delta"] - delta) < 1e-9
                      and not r.get("error")]

            if not subset:
                continue

            # Omega間のmedian中央値（Omega依存性の指標）
            omega_medians = [r["median_log10"] for r in subset]
            omega_stds    = [r["std_log10"]    for r in subset]
            # z内のstd平均（z依存性の指標）
            within_std    = float(np.mean(omega_stds))
            between_std   = float(np.std(omega_medians))

            # 全z結果をフラット化
            all_lerrs = [zr["log10_rel_err"]
                         for r in subset for zr in r["z_results"]]

            summary[key] = {
                "g": g, "delta": delta,
                "n_omega":      len(subset),
                "n_z_per_omega": N_Z,
                "n_total":      len(all_lerrs),
                # 全体統計
                "global_median": float(np.median(all_lerrs)),
                "global_mean":   float(np.mean(all_lerrs)),
                "global_std":    float(np.std(all_lerrs)),
                # 分散分解
                "within_omega_std":   within_std,   # z依存性の指標
                "between_omega_std":  between_std,  # Omega依存性の指標
                # Omega間のmedian分布
                "omega_median_min":   float(np.min(omega_medians)),
                "omega_median_max":   float(np.max(omega_medians)),
                "omega_median_range": float(np.max(omega_medians) - np.min(omega_medians)),
            }
    return summary

# ─────────────────────────────────────────
# 保存・読み込み
# ─────────────────────────────────────────
def save_results(records, summary, meta):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": records},
                  f, indent=2, ensure_ascii=False)

def load_results():
    if not Path(RESULT_PATH).exists():
        return []
    with open(RESULT_PATH) as f:
        return json.load(f).get("records", [])

# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() or 1))
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    setup_logging()
    t_start = time.perf_counter()

    total = len(G_LIST) * len(DELTAS) * N_OMEGA

    log.info("=" * 65)
    log.info("omega_fixed_scan.py  Omega固定・z変更 依存性分離スキャン")
    log.info(f"Python {sys.version.split()[0]}  workers={args.workers}")
    log.info(f"g={G_LIST}  N_cut={N_CUT}  delta={DELTAS}")
    log.info(f"N_omega={N_OMEGA}  N_z={N_Z}  総Omegaケース={total}")
    log.info(f"総theta計算数={total * N_Z:,}  (naive+s22)")
    log.info("=" * 65)

    all_tasks = [
        (g, delta, di, oi)
        for g            in G_LIST
        for di, delta    in enumerate(DELTAS)
        for oi           in range(N_OMEGA)
    ]

    # resume
    done_records = []
    done_keys    = set()
    if args.resume:
        done_records = load_results()
        for r in done_records:
            done_keys.add((r["g"], round(r["delta"], 1), r["omega_idx"]))
        log.info(f"resume: {len(done_keys)} Omega ケーススキップ")

    pending = [t for t in all_tasks
               if (t[0], round(t[1], 1), t[3]) not in done_keys]
    log.info(f"実行対象: {len(pending)} / {total} Omega ケース"
             f" ({len(pending) * N_Z:,} theta計算)")

    results   = list(done_records)
    completed = len(done_records)
    chunksize = max(1, len(pending) // (args.workers * 4))

    with multiprocessing.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(run_omega_case, pending,
                                      chunksize=chunksize):
            results.append(r)
            completed += 1

            if r.get("error"):
                log.warning(f"ERROR g={r['g']} delta={r['delta']} "
                            f"omega_idx={r['omega_idx']}: {r['error']}")
            elif completed % 10 == 0 or completed == total:
                pct = 100.0 * completed / total
                log.info(f"[{completed:3d}/{total}] ({pct:.1f}%) "
                         f"g={r['g']} δ={r['delta']:.1f} "
                         f"omega_idx={r['omega_idx']:2d} "
                         f"median={r.get('median_log10', float('nan')):.2f} "
                         f"std={r.get('std_log10', float('nan')):.2f}")

            if completed % 30 == 0:
                summary = compute_summary(results)
                meta = {"status": "running", "completed": completed,
                        "elapsed_s": round(time.perf_counter()-t_start, 1),
                        "g_list": G_LIST, "N_cut": N_CUT, "deltas": DELTAS,
                        "N_omega": N_OMEGA, "N_z": N_Z,
                        "workers": args.workers,
                        "timestamp": datetime.now().isoformat(timespec="seconds")}
                save_results(results, summary, meta)
                log.info(f"  → 中間保存: {completed} Omega ケース完了")

    elapsed = time.perf_counter() - t_start
    summary = compute_summary(results)
    meta = {
        "status": "completed", "completed": completed,
        "elapsed_s": round(elapsed, 1), "elapsed_min": round(elapsed/60, 2),
        "g_list": G_LIST, "N_cut": N_CUT, "deltas": DELTAS,
        "N_omega": N_OMEGA, "N_z": N_Z,
        "omega_seed_base": OMEGA_SEED_BASE, "z_seed_base": Z_SEED_BASE,
        "workers": args.workers,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_results(results, summary, meta)

    # コンソールサマリー
    log.info("=" * 65)
    log.info(f"完了: {completed} Omega ケース  {elapsed/60:.1f} 分")
    log.info("")
    log.info("【分散分解サマリー】")
    log.info("  within_std  = z を変えたときの誤差変動 (z依存性)")
    log.info("  between_std = Omega を変えたときの中央値変動 (Omega依存性)")
    log.info("")
    log.info(f"{'g':>4} {'delta':>6} {'global_med':>11} {'within_std':>11} "
             f"{'between_std':>12} {'omega_range':>12}")
    log.info("-" * 62)
    for g in G_LIST:
        for delta in DELTAS:
            s = summary.get(f"g{g}_d{delta:.1f}")
            if not s:
                continue
            log.info(f"{g:>4} {delta:>6.1f} {s['global_median']:>11.2f} "
                     f"{s['within_omega_std']:>11.2f} "
                     f"{s['between_omega_std']:>12.2f} "
                     f"{s['omega_median_range']:>12.2f}")
        log.info("")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
