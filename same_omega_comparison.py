"""
same_omega_comparisonPlus.py

「同一Ω比較」: naive vs s22 を同一の Ω, z で評価し、
値の一致（相対誤差）と計算時間を比較する。

対象: g=8, 9  (N_cut=1, 2)
S_{(2,2)} 上の点 (ε_{12}=0 が厳密に成立する領域) と
S_{(2,2)} から外れた点 (ε_{12}≠0) の両方をテスト。

機能追加:
  - 途中保存 (results.json) と再開 (resume) 機能
  - 進捗ログ (progress_log.txt)
  - config.txt による設定変更

実行方法:
    python same_omega_comparisonPlus.py

依存: numpy のみ (scipy は任意)
"""

import time
import json
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_comparison.txt")
LOG_PATH    = os.path.join(SCRIPT_DIR, "comparison_log.txt")
RESULT_PATH = os.path.join(SCRIPT_DIR, "comparison_results.json")

# ─────────────────────────────────────────────
# 設定ファイル
# ─────────────────────────────────────────────

def load_config():
    config = {
        "g_list":     [2,3,4,5,6,7,8, 9, 10, 11, 12,13,14,15,16,17],
        "N_cut_list": [1, 2],
        "delta":      0.5,
        "seed":       42,
        "resume":     True,
    }
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("# same_omega_comparisonPlus config\n")
            f.write("#\n")
            f.write("# g_list     : カンマ区切りの次元リスト\n")
            f.write("# N_cut_list : カンマ区切りの打ち切り幅リスト\n")
            f.write("# delta      : S22外れ点の摂動量 (0=S22上, 1=一般点)\n")
            f.write("# seed       : 乱数シード\n")
            f.write("# resume     : true で前回の途中から再開\n")
            f.write("#\n")
            f.write("g_list = 8,9,10,11,12\n")
            f.write("N_cut_list = 1,2\n")
            f.write("delta = 0.5\n")
            f.write("seed = 42\n")
            f.write("resume = true\n")
        print(f"config_comparison.txt を作成しました: {CONFIG_PATH}")
        return config

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key == "g_list":
                config["g_list"] = [int(x) for x in val.split(",")]
            elif key == "N_cut_list":
                config["N_cut_list"] = [int(x) for x in val.split(",")]
            elif key == "delta":
                config["delta"] = float(val)
            elif key == "seed":
                config["seed"] = int(val)
            elif key == "resume":
                config["resume"] = val.lower() == "true"
    return config

# ─────────────────────────────────────────────
# ログ
# ─────────────────────────────────────────────

def log(msg, also_print=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ─────────────────────────────────────────────
# Resume (途中保存・再開)
# ─────────────────────────────────────────────

def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_result(entry, all_results):
    all_results.append(entry)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

def already_done(all_results, g, N_cut, label):
    """同一の (g, N_cut, label) の結果が既存か確認"""
    return any(
        r["g"] == g and r["N_cut"] == N_cut and r["label"] == label
        for r in all_results
    )

# ─────────────────────────────────────────────
# 推定完了時刻
# ─────────────────────────────────────────────

def format_eta(eta_seconds):
    eta_sec = int(eta_seconds)
    h = eta_sec // 3600
    m = (eta_sec % 3600) // 60
    s = eta_sec % 60
    finish_at = datetime.now() + timedelta(seconds=eta_seconds)
    finish_str = finish_at.strftime("%Y-%m-%d %H:%M:%S")
    return f"{h:02d}:{m:02d}:{s:02d}", finish_str

# ─────────────────────────────────────────────
# 1. Theta 関数の実装
# ─────────────────────────────────────────────

def theta_naive(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    """
    g 次元 Siegel theta 関数の打ち切り級数 (naive).

    θ(z|Ω) = Σ_{n ∈ Z^g, |n_i|≤N_cut} exp(iπ n^T Ω n + 2πi n^T z)
    """
    g = len(z)
    ns = np.arange(-N_cut, N_cut + 1)

    grids = np.meshgrid(*[ns] * g, indexing='ij')
    lattice = np.stack([gr.ravel() for gr in grids], axis=1).astype(float)

    quad = np.einsum('ki,ij,kj->k', lattice, Omega, lattice)
    lin  = lattice @ z

    log_terms = 1j * np.pi * quad + 2j * np.pi * lin
    log_terms_re = log_terms.real
    shift = log_terms_re.max()
    terms = np.exp(log_terms - shift)
    return np.exp(shift) * terms.sum()


def theta_s22(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    """
    S_{(2,2)} 分解による近似 theta 関数.

    θ(z|Ω) ≈ θ(z1|Ω1) × θ(z2|Ω2)  (off-diag ブロックを無視)
    """
    g = len(z)
    g1 = g // 2

    z1, z2 = z[:g1], z[g1:]
    Omega1  = Omega[:g1, :g1]
    Omega2  = Omega[g1:, g1:]

    val1 = theta_naive(z1, Omega1, N_cut)
    val2 = theta_naive(z2, Omega2, N_cut)
    return val1 * val2


# ─────────────────────────────────────────────
# 2. テスト用 Ω, z の生成
# ─────────────────────────────────────────────

def make_omega_on_S22(g: int, rng: np.random.Generator) -> np.ndarray:
    """S_{(2,2)} 上の周期行列: off-diag ブロック = 0"""
    g1 = g // 2

    def make_block(size):
        A = rng.standard_normal((size, size)) * 0.3
        return 1j * (A @ A.T + np.eye(size) * 1.5)

    Omega = np.zeros((g, g), dtype=complex)
    Omega[:g1, :g1] = make_block(g1)
    Omega[g1:, g1:] = make_block(g - g1)
    return (Omega + Omega.T) / 2


def make_omega_off_S22(g: int, delta: float,
                       rng: np.random.Generator) -> np.ndarray:
    """S_{(2,2)} から delta だけ外れた周期行列"""
    Omega = make_omega_on_S22(g, rng)
    g1 = g // 2
    g2 = g - g1
    perturb = 1j * rng.standard_normal((g1, g2)) * 0.3 * delta
    Omega[:g1, g1:] = perturb
    Omega[g1:, :g1] = perturb.T.conj()
    return (Omega + Omega.T) / 2


def make_z(g: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(g) * 0.5 + 1j * rng.standard_normal(g) * 0.1


# ─────────────────────────────────────────────
# 3. 比較実験
# ─────────────────────────────────────────────

def relative_error(v_naive: complex, v_s22: complex) -> float:
    denom = abs(v_naive)
    if denom < 1e-300:
        return float('nan')
    return abs(v_naive - v_s22) / denom


def run_comparison(g: int, N_cut: int, Omega: np.ndarray, z: np.ndarray,
                   label: str) -> dict:
    t0 = time.perf_counter()
    val_naive = theta_naive(z, Omega, N_cut)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_s22 = theta_s22(z, Omega, N_cut)
    t_s22 = time.perf_counter() - t0

    rel_err = relative_error(val_naive, val_s22)
    speedup = t_naive / t_s22 if t_s22 > 0 else float('inf')

    return {
        'g': g, 'N_cut': N_cut, 'label': label,
        'val_naive_re': val_naive.real,
        'val_naive_im': val_naive.imag,
        'val_s22_re':   val_s22.real,
        'val_s22_im':   val_s22.imag,
        'rel_err': rel_err,
        't_naive': t_naive,
        't_s22':   t_s22,
        'speedup': speedup,
        'timestamp': datetime.now().isoformat(),
    }


def print_result(r: dict):
    val_naive = complex(r['val_naive_re'], r['val_naive_im'])
    val_s22   = complex(r['val_s22_re'],   r['val_s22_im'])
    print(f"  g={r['g']}, N_cut={r['N_cut']}, [{r['label']}]")
    print(f"    naive : {val_naive:.6f}   ({r['t_naive']:.4f}s)")
    print(f"    s22   : {val_s22:.6f}   ({r['t_s22']:.4f}s)")
    print(f"    相対誤差: {r['rel_err']:.2e}")
    print(f"    高速化倍率: {r['speedup']:.1f}x")


# ─────────────────────────────────────────────
# 4. サマリー表示
# ─────────────────────────────────────────────

def print_summary(all_results: list):
    log("=" * 65)
    log("サマリー表 (同一Ω・z での比較)")
    log("=" * 65)
    header = (f"{'g':>3} {'N':>2} {'ラベル':<18} "
              f"{'相対誤差':>12} {'naive(s)':>10} {'s22(s)':>10} {'倍率':>8}")
    log(header)
    log("-" * 65)
    for r in sorted(all_results, key=lambda x: (x['g'], x['N_cut'], x['label'])):
        label_short = "S22上" if "S22上" in r['label'] else "S22外"
        log(f"{r['g']:>3} {r['N_cut']:>2} {label_short:<18} "
            f"{r['rel_err']:>12.2e} "
            f"{r['t_naive']:>10.4f} "
            f"{r['t_s22']:>10.4f} "
            f"{r['speedup']:>8.1f}x")
    log("")
    log("解釈:")
    log("  S22上 → 相対誤差≈0 が期待値 (Fay公式により ε_{12}=0 厳密)")
    log("  S22外 → 相対誤差 > 0 が期待値 (off-diag 無視による近似誤差)")
    log("  高速化倍率 = naive時間 / s22時間")
    log(f"結果ファイル: {RESULT_PATH}")
    log(f"ログファイル: {LOG_PATH}")


# ─────────────────────────────────────────────
# 5. メイン
# ─────────────────────────────────────────────

def main():
    config      = load_config()
    all_results = load_results() if config["resume"] else []

    rng = np.random.default_rng(seed=config["seed"])

    g_list     = config["g_list"]
    N_cut_list = config["N_cut_list"]
    delta      = config["delta"]

    # ケースラベル定義
    LABEL_ON  = "S22上 (ε=0厳密)"
    LABEL_OFF = f"S22外 (delta={delta})"

    # 全ケース数 = g × N_cut × 2 (on/off)
    total_cases = len(g_list) * len(N_cut_list) * 2
    case_num = 0

    log("=" * 65)
    log("同一 Ω 比較: naive vs s22  開始")
    log(f"g_list={g_list}")
    log(f"N_cut_list={N_cut_list}")
    log(f"delta={delta}  seed={config['seed']}  resume={config['resume']}")
    log(f"既完了ケース数: {len(all_results)}")
    log("=" * 65)

    try:
        for g in g_list:
            # g ごとに同一の z, Ω を使う（再現性のため rng を固定）
            rng_g = np.random.default_rng(seed=config["seed"] + g)
            z        = make_z(g, rng_g)
            Omega_on = make_omega_on_S22(g,          rng_g)
            Omega_off = make_omega_off_S22(g, delta, rng_g)

            log(f"\n─── g={g} ───")

            for N_cut in N_cut_list:
                log(f"\n  [N_cut={N_cut}]")

                # ── ケース A: S22上 ──
                case_num += 1
                pct = 100.0 * case_num / total_cases
                if config["resume"] and already_done(
                        all_results, g, N_cut, LABEL_ON):
                    log(f"  スキップ [{case_num}/{total_cases}] "
                        f"g={g} N={N_cut} {LABEL_ON}")
                    # スキップ分を表示だけする
                    cached = next(r for r in all_results
                                  if r["g"] == g and r["N_cut"] == N_cut
                                  and r["label"] == LABEL_ON)
                    print_result(cached)
                else:
                    log(f"  実行 [{case_num}/{total_cases}] "
                        f"({pct:.1f}%) g={g} N={N_cut} {LABEL_ON}")
                    r_on = run_comparison(g, N_cut, Omega_on, z, LABEL_ON)
                    print_result(r_on)
                    save_result(r_on, all_results)
                    log(f"  保存完了: g={g} N={N_cut} {LABEL_ON} "
                        f"rel_err={r_on['rel_err']:.2e}")

                # ── ケース B: S22外 ──
                case_num += 1
                pct = 100.0 * case_num / total_cases
                if config["resume"] and already_done(
                        all_results, g, N_cut, LABEL_OFF):
                    log(f"  スキップ [{case_num}/{total_cases}] "
                        f"g={g} N={N_cut} {LABEL_OFF}")
                    cached = next(r for r in all_results
                                  if r["g"] == g and r["N_cut"] == N_cut
                                  and r["label"] == LABEL_OFF)
                    print_result(cached)
                else:
                    log(f"  実行 [{case_num}/{total_cases}] "
                        f"({pct:.1f}%) g={g} N={N_cut} {LABEL_OFF}")
                    r_off = run_comparison(g, N_cut, Omega_off, z, LABEL_OFF)
                    print_result(r_off)
                    save_result(r_off, all_results)
                    log(f"  保存完了: g={g} N={N_cut} {LABEL_OFF} "
                        f"rel_err={r_off['rel_err']:.2e}")

    except KeyboardInterrupt:
        log("\n中断されました（KeyboardInterrupt）")
        log("resume=true で再実行すると続きから再開できます")
        print_summary(all_results)
        sys.exit(0)

    except Exception as e:
        log(f"エラー発生: {e}")
        raise

    print_summary(all_results)


if __name__ == "__main__":
    main()