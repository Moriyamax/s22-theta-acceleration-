import numpy as np
import time
import json
import os
import sys
from datetime import datetime, timedelta
from itertools import product as iproduct

# ============================================================
# パス設定
# ============================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.txt")
LOG_PATH    = os.path.join(SCRIPT_DIR, "progress_log.txt")
RESULT_PATH = os.path.join(SCRIPT_DIR, "results.json")

# ============================================================
# 実行順序
# ============================================================
MODE_ORDER = ["s22", "naive"]

# ============================================================
# 設定ファイル
# ============================================================
def load_config():
    config = {
        "g_list":       [5,6,7,8,9,10,11,12,13,14,15,16,17],
        "N_cut_list":   [1, 2],
        "mode":         "all",
        "seed":         42,
        "report_every": 1_000_000,
        "resume":       True,
    }
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("# Theta function benchmark config\n")
            f.write("#\n")
            f.write("# mode の選択肢:\n")
            f.write("#   s22   : S_{(2,2)}分解（高速）\n")
            f.write("#   naive : 一般点 打ち切り級数（比較基準）\n")
            f.write("#   all   : 上記を s22→naive の順に実行\n")
            f.write("#\n")
            f.write("g_list = 5,6,7,8,9,10,11,12,13,14,15,16,17\n")
            f.write("N_cut_list = 1,2\n")
            f.write("mode = all\n")
            f.write("seed = 42\n")
            f.write("report_every = 1000000\n")
            f.write("resume = true\n")
        print(f"config.txt を作成しました: {CONFIG_PATH}")
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
            elif key == "mode":
                config["mode"] = val
            elif key == "seed":
                config["seed"] = int(val)
            elif key == "report_every":
                config["report_every"] = int(val)
            elif key == "resume":
                config["resume"] = val.lower() == "true"
    return config

# ============================================================
# ログ
# ============================================================
def log(msg, also_print=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ============================================================
# Resume
# ============================================================
def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_result(entry, all_results):
    all_results.append(entry)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

def already_done(all_results, mode, g, N_cut):
    return any(
        r["mode"] == mode and r["g"] == g and r["N_cut"] == N_cut
        for r in all_results
    )

# ============================================================
# 推定完了時刻
# ============================================================
def format_eta(eta_seconds):
    eta_sec = int(eta_seconds)
    h = eta_sec // 3600
    m = (eta_sec % 3600) // 60
    s = eta_sec % 60
    finish_at = datetime.now() + timedelta(seconds=eta_seconds)
    finish_str = finish_at.strftime("%Y-%m-%d %H:%M:%S")
    return f"{h:02d}:{m:02d}:{s:02d}", finish_str

# ============================================================
# 周期行列生成
# ============================================================
def make_omega(g, seed=42):
    """
    Siegel上半空間の元を生成する：対称かつIm(Ω)が正定値。
    実験目的のランダム点であり、実際のHitchin系周期行列ではない。
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((g, g))
    im = A @ A.T + g * np.eye(g)   # 正定値対称行列
    re = rng.standard_normal((g, g))
    re = (re + re.T) / 2            # 対称化
    return re + 1j * im

def make_block_omegas(g_total, seed=42):
    """
    S_{(2,2)}分解用：g_total を2つのブロックに分割した周期行列を返す。
    g_total が奇数の場合は (g_total//2, g_total//2+1) に分割。
    例: g=17 → g1=8, g2=9

    注意: この分割はS_{(2,2)}のblock-diagonal構造に着想を得た
    実装上の算術的選択であり、Prym多様体の幾何学的分解と
    必ずしも対応するものではない（付録F参照）。
    """
    g1 = g_total // 2
    g2 = g_total - g1
    Omega1 = make_omega(g1, seed=seed)
    Omega2 = make_omega(g2, seed=seed + 1)
    return Omega1, Omega2, g1, g2

# ============================================================
# naive Theta計算（進捗表示付き）
# ============================================================
def theta_naive(z, Omega, N_cut, label="", report_every=1_000_000):
    """
    打ち切り級数によるTheta関数計算。
    θ(z|Ω) = Σ_{n∈Z^g, |n_i|≤N_cut} exp(πi nᵀΩn + 2πi nᵀz)
    """
    g = len(z)
    total_terms = (2 * N_cut + 1) ** g
    indices = range(-N_cut, N_cut + 1)

    result = 0.0 + 0.0j
    start = time.perf_counter()

    for count, n in enumerate(iproduct(indices, repeat=g)):
        n_arr = np.array(n, dtype=float)
        phase = (np.pi * 1j * (n_arr @ Omega @ n_arr)
                 + 2 * np.pi * 1j * (n_arr @ z))
        result += np.exp(phase)

        if count > 0 and count % report_every == 0:
            elapsed = time.perf_counter() - start
            pct = 100.0 * count / total_terms
            eta_sec = elapsed / (pct / 100.0) - elapsed
            eta_str, finish_str = format_eta(eta_sec)
            log(
                f"{label} | "
                f"{pct:.3f}% ({count:,}/{total_terms:,}) | "
                f"経過:{elapsed:.0f}s | "
                f"残り:{eta_str} | "
                f"推定完了:{finish_str}"
            )

    return result, time.perf_counter() - start

# ============================================================
# 2モードの実行
# ============================================================
def run_s22(g, N_cut, seed, report_every):
    """
    S_{(2,2)}分解モード：g_total次元の計算をg1+g2に分割して実行。
    θ(z|Ω) ≈ θ_1(z_1|Ω_1) × θ_2(z_2|Ω_2)
    """
    Omega1, Omega2, g1, g2 = make_block_omegas(g, seed)
    z1 = np.random.default_rng(seed).standard_normal(g1)
    z2 = np.random.default_rng(seed+1).standard_normal(g2)

    label1 = f"s22-block1 g={g}(g1={g1}) N={N_cut}"
    label2 = f"s22-block2 g={g}(g2={g2}) N={N_cut}"

    start = time.perf_counter()
    t1, _ = theta_naive(z1, Omega1, N_cut,
                        label=label1, report_every=report_every)
    t2, _ = theta_naive(z2, Omega2, N_cut,
                        label=label2, report_every=report_every)
    result = t1 * t2
    return result, time.perf_counter() - start

def run_naive(g, N_cut, seed, report_every):
    """
    naiveモード：一般点Ωに対して打ち切り級数を直接計算（比較基準）。
    """
    Omega = make_omega(g, seed)
    z = np.random.default_rng(seed).standard_normal(g)
    label = f"naive g={g} N={N_cut}"
    return theta_naive(z, Omega, N_cut,
                       label=label, report_every=report_every)

RUNNERS = {
    "s22":   run_s22,
    "naive": run_naive,
}

# ============================================================
# 完了サマリー
# ============================================================
def print_summary(all_results):
    log("=" * 60)
    log("完了サマリー")
    log("=" * 60)

    for mode in MODE_ORDER:
        rows = [r for r in all_results if r["mode"] == mode]
        if not rows:
            continue
        log(f"--- {mode} ---")
        for r in sorted(rows, key=lambda x: (x["N_cut"], x["g"])):
            n_terms = r.get("n_terms", "N/A")
            n_str = f"{n_terms:.2e}" if isinstance(n_terms, int) else n_terms
            log(f"  g={r['g']:2d} N={r['N_cut']} "
                f"項数:{n_str}  時間:{r['time_s']:.4f}秒")

    # 削減比の表示
    log("--- 削減比（naive/s22）---")
    for N_cut in [1, 2]:
        for g in sorted(set(r["g"] for r in all_results)):
            naive_row = next((r for r in all_results
                              if r["mode"] == "naive"
                              and r["g"] == g
                              and r["N_cut"] == N_cut), None)
            s22_row   = next((r for r in all_results
                              if r["mode"] == "s22"
                              and r["g"] == g
                              and r["N_cut"] == N_cut), None)
            if naive_row and s22_row:
                ratio = naive_row["time_s"] / s22_row["time_s"]
                log(f"  g={g:2d} N={N_cut}: {ratio:.1f}倍")

    log("=" * 60)
    log(f"結果: {RESULT_PATH}")
    log(f"ログ: {LOG_PATH}")

# ============================================================
# メイン
# ============================================================
def main():
    config      = load_config()
    all_results = load_results() if config["resume"] else []

    mode = config["mode"]
    if mode == "all":
        modes_to_run = MODE_ORDER   # ["s22", "naive"]
    elif mode in RUNNERS:
        modes_to_run = [mode]
    else:
        log(f"不明なmode: {mode}  使用可能: {list(RUNNERS.keys()) + ['all']}")
        sys.exit(1)

    log("=" * 60)
    log("ベンチマーク開始")
    log(f"実行モード: {modes_to_run}")
    log(f"g_list={config['g_list']}")
    log(f"N_cut_list={config['N_cut_list']}")
    log(f"resume={config['resume']}")
    log(f"既完了ケース数: {len(all_results)}")
    log("=" * 60)

    g_list       = config["g_list"]
    N_cut_list   = config["N_cut_list"]
    seed         = config["seed"]
    report_every = config["report_every"]

    total_cases = len(modes_to_run) * len(g_list) * len(N_cut_list)
    case_num    = 0

    for current_mode in modes_to_run:
        log(f"{'='*20} mode={current_mode} 開始 {'='*20}")

        for g in g_list:
            for N_cut in N_cut_list:
                case_num += 1
                pct_case = 100.0 * case_num / total_cases

                if config["resume"] and already_done(
                        all_results, current_mode, g, N_cut):
                    log(f"スキップ [{case_num}/{total_cases}] "
                        f"mode={current_mode} g={g} N={N_cut}")
                    continue

                log(f"開始 [{case_num}/{total_cases}] "
                    f"({pct_case:.1f}%) "
                    f"mode={current_mode} g={g} N={N_cut}")

                try:
                    val, elapsed = RUNNERS[current_mode](
                        g, N_cut, seed, report_every)

                    entry = {
                        "mode":      current_mode,
                        "g":         g,
                        "N_cut":     N_cut,
                        "time_s":    round(elapsed, 6),
                        "n_terms":   (2*N_cut+1)**g,
                        "timestamp": datetime.now().isoformat(),
                    }
                    save_result(entry, all_results)
                    log(f"完了 mode={current_mode} "
                        f"g={g} N={N_cut} → {elapsed:.4f}秒")

                except KeyboardInterrupt:
                    log("中断されました（KeyboardInterrupt）")
                    log(f"中断時点: mode={current_mode} g={g} N={N_cut}")
                    log("resume=true で再実行すると続きから再開できます")
                    print_summary(all_results)
                    sys.exit(0)

                except Exception as e:
                    log(f"エラー mode={current_mode} g={g} N={N_cut}: {e}")
                    continue

        log(f"{'='*20} mode={current_mode} 完了 {'='*20}")

    print_summary(all_results)

if __name__ == "__main__":
    main()