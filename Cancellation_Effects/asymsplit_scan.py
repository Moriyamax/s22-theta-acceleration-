"""
asymsplit_scan.py  ― 非対称分割 (g1, g2) による誤差比較スキャン

S_{(2,2)} 因子化の分割パターン (g1, g2) を変えたとき、
誤差分布がどう変わるかを系統的に測定する。

設定:
  G_LIST   = [8, 10, 12]
  N_CUT    = 1
  DELTAS   = 0.1〜1.0 (10点)
  N_SEED   = 100
  splits   = g=8: (1,7),(2,6),(3,5),(4,4)
             g=10: (1,9),(2,8),(3,7),(4,6),(5,5)
             g=12: (1,11),(2,10),(3,9),(4,8),(5,7),(6,6)

最適化:
  同一 (g, delta, seed) の naive値をキャッシュして再利用。
  naive計算は (g, delta, seed) ごとに1回のみ。

定義式・行列生成:
  appendix_i_plots.py / cancellation_scan.py と完全一致。

出力:
  asymsplit_results.json
  asymsplit_scan.log

実行方法:
  python asymsplit_scan.py
  python asymsplit_scan.py --workers 6 --g 8,10   # g指定
  python asymsplit_scan.py --resume
"""

import numpy as np
import json, time, os, sys, argparse, logging, multiprocessing
from itertools import product as iproduct
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
G_LIST    = [8, 10, 12]
N_CUT     = 1
DELTAS    = [round(0.1 * k, 1) for k in range(1, 11)]
N_SEED    = 100
SEED_BASE = 3000   # cancellation_scan(2000)と区別

SPLITS = {
    8:  [(1,7),(2,6),(3,5),(4,4)],
    10: [(1,9),(2,8),(3,7),(4,6),(5,5)],
    12: [(1,11),(2,10),(3,9),(4,8),(5,7),(6,6)],
}

RESULT_PATH = "asymsplit_results.json"
LOG_PATH    = "asymsplit_scan.log"
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

    z = rng.standard_normal(g)

    eigs = np.linalg.eigvalsh(Omega.imag)
    assert np.all(eigs > 0), \
        f"Im(Omega) not PD: min={eigs.min():.3e} (g={g},delta={delta},seed={seed})"

    return Omega, z

# ─────────────────────────────────────────
# theta 関数 (cancellation_scan.py と同一)
# ─────────────────────────────────────────
def theta_naive(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    result = 0j
    it  = iproduct(range(-N_cut, N_cut + 1), repeat=len(z))
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


def theta_s22_asym(z: np.ndarray, Omega: np.ndarray,
                   g1: int, N_cut: int) -> complex:
    """任意の分割 g1 + g2 = g での因子化"""
    v1 = theta_naive(z[:g1],     Omega[:g1, :g1],         N_cut)
    v2 = theta_naive(z[g1:],     Omega[g1:, g1:],         N_cut)
    return v1 * v2

# ─────────────────────────────────────────
# Worker: (g, delta, seed_i) を受け取り
#         naive + 全分割パターンの s22 を計算して返す
# ─────────────────────────────────────────
def run_case(args):
    g, delta, delta_idx, seed_i, splits_g = args
    seed = SEED_BASE + g * 10000 + delta_idx * 100 + seed_i

    try:
        Omega, z = make_omega(g, delta, seed)
    except AssertionError as e:
        return {"g": g, "delta": delta, "seed": seed, "seed_i": seed_i,
                "error": str(e), "splits": {}}

    # naive (全分割共通の真値)
    vn = theta_naive(z, Omega, N_CUT)

    # 各分割パターンの s22
    split_results = {}
    for g1 in splits_g:
        g2  = g - g1
        vs  = theta_s22_asym(z, Omega, g1, N_CUT)
        rel = float(abs(vn - vs) / abs(vn)) if abs(vn) > 1e-300 else float("nan")
        log_rel = float(np.log10(rel)) if (not np.isnan(rel) and rel > 0) else -16.0
        split_results[f"{g1}_{g2}"] = {
            "g1": g1, "g2": g2,
            "rel_err":       rel,
            "log10_rel_err": log_rel,
            "val_s22_re":    float(vs.real),
            "val_s22_im":    float(vs.imag),
        }

    # 補助統計
    g1_half = g // 2
    offblock_norm = float(
        np.linalg.norm(Omega.imag[:g1_half, g1_half:])
        / np.sqrt(g1_half * (g - g1_half))
    ) if delta > 0.0 else 0.0

    return {
        "g":             g,
        "delta":         delta,
        "seed":          seed,
        "seed_i":        seed_i,
        "offblock_norm": offblock_norm,
        "val_naive_re":  float(vn.real),
        "val_naive_im":  float(vn.imag),
        "splits":        split_results,
    }

# ─────────────────────────────────────────
# サマリー統計
# ─────────────────────────────────────────
def compute_summary(records, g_list):
    summary = {}
    for g in g_list:
        splits_g = [g1 for g1, _ in SPLITS[g]]
        for delta in DELTAS:
            for g1 in splits_g:
                g2  = g - g1
                key = f"g{g}_d{delta:.1f}_s{g1}_{g2}"
                subset = [
                    r["splits"][f"{g1}_{g2}"]["log10_rel_err"]
                    for r in records
                    if r["g"] == g
                    and abs(r["delta"] - delta) < 1e-9
                    and f"{g1}_{g2}" in r.get("splits", {})
                    and not np.isnan(r["splits"][f"{g1}_{g2}"]["log10_rel_err"])
                ]
                summary[key] = {
                    "g": g, "delta": delta, "g1": g1, "g2": g2,
                    "n":             len(subset),
                    "median_log10":  float(np.median(subset))      if subset else float("nan"),
                    "mean_log10":    float(np.mean(subset))        if subset else float("nan"),
                    "std_log10":     float(np.std(subset))         if subset else float("nan"),
                    "min_log10":     float(np.min(subset))         if subset else float("nan"),
                    "max_log10":     float(np.max(subset))         if subset else float("nan"),
                    "q25_log10":     float(np.percentile(subset, 25)) if subset else float("nan"),
                    "q75_log10":     float(np.percentile(subset, 75)) if subset else float("nan"),
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
    parser.add_argument("--g",       type=str, default=None,
                        help="実行するgをカンマ区切りで指定 例: --g 8,10")
    args = parser.parse_args()

    g_list = [int(x) for x in args.g.split(",")] if args.g else G_LIST
    # SPLITS に存在するgのみ
    g_list = [g for g in g_list if g in SPLITS]

    setup_logging()
    t_start = time.perf_counter()

    log.info("=" * 65)
    log.info("asymsplit_scan.py  非対称分割誤差スキャン")
    log.info(f"Python {sys.version.split()[0]}  workers={args.workers}")
    log.info(f"g={g_list}  N_cut={N_CUT}  delta={DELTAS}  N_seed={N_SEED}")
    for g in g_list:
        log.info(f"  g={g} splits: {SPLITS[g]}")

    # 全タスク: (g, delta, delta_idx, seed_i, splits_g)
    all_tasks = [
        (g, delta, di, si, [g1 for g1, _ in SPLITS[g]])
        for g            in g_list
        for di, delta    in enumerate(DELTAS)
        for si           in range(N_SEED)
    ]
    total = len(all_tasks)
    log.info(f"総タスク数: {total} (g×delta×seed = {len(g_list)}×{len(DELTAS)}×{N_SEED})")
    log.info("=" * 65)

    # resume
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

    results   = list(done_records)
    completed = len(done_records)
    chunksize = max(1, len(pending) // (args.workers * 4))

    with multiprocessing.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(run_case, pending, chunksize=chunksize):
            results.append(r)
            completed += 1

            if "error" in r:
                log.warning(f"ERROR g={r['g']} delta={r['delta']} "
                            f"seed_i={r['seed_i']}: {r['error']}")
            elif completed % 100 == 0 or completed == total:
                pct = 100.0 * completed / total
                # 対称分割(等分)の誤差を代表値として表示
                g     = r["g"]
                g_sym = str(g // 2) + "_" + str(g - g // 2)
                sym   = r["splits"].get(g_sym, {})
                log.info(f"[{completed}/{total}] ({pct:.1f}%) "
                         f"g={g} δ={r['delta']:.1f} seed_i={r['seed_i']:3d} "
                         f"sym_log10={sym.get('log10_rel_err', float('nan')):.2f}")

            if completed % 500 == 0:
                summary = compute_summary(results, g_list)
                meta = {"status": "running", "completed": completed,
                        "elapsed_s": round(time.perf_counter() - t_start, 1),
                        "g_list": g_list, "N_cut": N_CUT, "deltas": DELTAS,
                        "N_seed": N_SEED, "seed_base": SEED_BASE,
                        "workers": args.workers,
                        "timestamp": datetime.now().isoformat(timespec="seconds")}
                save_results(results, summary, meta)
                log.info(f"  → 中間保存: {completed} 件")

    elapsed = time.perf_counter() - t_start
    summary = compute_summary(results, g_list)
    meta = {"status": "completed", "completed": completed,
            "elapsed_s": round(elapsed, 1), "elapsed_min": round(elapsed/60, 2),
            "g_list": g_list, "N_cut": N_CUT, "deltas": DELTAS,
            "N_seed": N_SEED, "seed_base": SEED_BASE,
            "workers": args.workers,
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    save_results(results, summary, meta)

    # コンソールサマリー: 分割パターン別 median (delta平均)
    log.info("=" * 65)
    log.info(f"完了: {completed} ケース  {elapsed/60:.1f} 分")
    log.info("")
    log.info(f"{'g':>4} {'split':>8} {'median(δ-avg)':>14} {'mean':>8} {'std':>6}")
    log.info("-" * 50)
    for g in g_list:
        for g1, g2 in SPLITS[g]:
            medians = [
                summary[f"g{g}_d{d:.1f}_s{g1}_{g2}"]["median_log10"]
                for d in DELTAS
                if not np.isnan(summary[f"g{g}_d{d:.1f}_s{g1}_{g2}"]["median_log10"])
            ]
            means = [
                summary[f"g{g}_d{d:.1f}_s{g1}_{g2}"]["mean_log10"]
                for d in DELTAS
                if not np.isnan(summary[f"g{g}_d{d:.1f}_s{g1}_{g2}"]["mean_log10"])
            ]
            stds = [
                summary[f"g{g}_d{d:.1f}_s{g1}_{g2}"]["std_log10"]
                for d in DELTAS
                if not np.isnan(summary[f"g{g}_d{d:.1f}_s{g1}_{g2}"]["std_log10"])
            ]
            sym = "←等分" if g1 == g2 else ""
            log.info(f"{g:>4} ({g1},{g2}){'':<3} "
                     f"{np.mean(medians):>14.2f} "
                     f"{np.mean(means):>8.2f} "
                     f"{np.mean(stds):>6.2f}  {sym}")
        log.info("")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
