"""
omega_fixed_scan2.py  ― g=10異常追加調査

調査内容:
  g=9,11: g=10前後での between_std の変化を確認
  g=10:   N_omega=10→50に増やし between_std=0が偶然でないか確認
  g=12:   N_omega=10 追加

設定:
  G_CONFIGS = {9: 10, 10: 40(追加), 11: 10, 12: 10}
  DELTAS    = [0.1, 0.5, 1.0]
  N_Z       = 100
  OMEGA_SEED_BASE = 7000  (既存5000と区別)

既存omega_fixed_scan.pyからのコピー、G_LIST/N_OMEGA部分のみ変更。
"""

import numpy as np
import json, time, os, sys, argparse, logging, multiprocessing
from itertools import product as iproduct
from datetime import datetime
from pathlib import Path

# ─── 設定 ───────────────────────────────
# g: (omega_idx_start, omega_idx_end) — 既存分と重ならないよう管理
G_CONFIGS = {
    9:  (0,  10),   # 新規 10個
    10: (10, 50),   # 既存10個の続き、追加40個
    11: (0,  10),   # 新規 10個
    12: (0,  10),   # 新規 10個
}
N_CUT     = 1
DELTAS    = [0.1, 0.5, 1.0]
N_Z       = 100
OMEGA_SEED_BASE = 7000
Z_SEED_BASE     = 8000

RESULT_PATH = "omega_fixed2_results.json"
LOG_PATH    = "omega_fixed2_scan.log"
CHUNK_SIZE  = 200_000

def setup_logging():
    logging.basicConfig(level=logging.INFO,
        format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])

log = logging.getLogger(__name__)

def make_omega(g, delta, seed):
    rng = np.random.default_rng(seed)
    g1, g2 = g // 2, g - g // 2
    A = rng.standard_normal((g1,g1)); Im1 = A@A.T + g1*np.eye(g1)
    Re1 = rng.standard_normal((g1,g1)); Re1=(Re1+Re1.T)/2
    B = rng.standard_normal((g2,g2)); Im2 = B@B.T + g2*np.eye(g2)
    Re2 = rng.standard_normal((g2,g2)); Re2=(Re2+Re2.T)/2
    Omega = np.zeros((g,g), dtype=complex)
    Omega[:g1,:g1] = Re1 + 1j*Im1
    Omega[g1:,g1:] = Re2 + 1j*Im2
    if delta > 0:
        C = delta*(rng.standard_normal((g1,g2)) + 1j*rng.standard_normal((g1,g2)))
        Omega[:g1,g1:]=C; Omega[g1:,:g1]=C.conj().T
    eigs = np.linalg.eigvalsh(Omega.imag)
    assert np.all(eigs>0), f"not PD g={g} d={delta} seed={seed}"
    return Omega

def theta_naive(z, Omega, N_cut):
    result=0j; buf=[]
    for n in iproduct(range(-N_cut,N_cut+1), repeat=len(z)):
        buf.append(n)
        if len(buf)>=CHUNK_SIZE:
            arr=np.array(buf,dtype=np.float64)
            result+=np.sum(np.exp(np.pi*1j*np.einsum('ij,ij->i',arr@Omega,arr)
                                  +2*np.pi*1j*(arr@z))); buf=[]
    if buf:
        arr=np.array(buf,dtype=np.float64)
        result+=np.sum(np.exp(np.pi*1j*np.einsum('ij,ij->i',arr@Omega,arr)
                              +2*np.pi*1j*(arr@z)))
    return result

def theta_s22(z, Omega, N_cut):
    g1=len(z)//2
    return (theta_naive(z[:g1],Omega[:g1,:g1],N_cut)*
            theta_naive(z[g1:],Omega[g1:,g1:],N_cut))

def run_omega_case(args):
    g, delta, delta_idx, omega_idx = args
    seed = OMEGA_SEED_BASE + g*10000 + delta_idx*100 + omega_idx
    try:
        Omega = make_omega(g, delta, seed)
    except AssertionError as e:
        return {"g":g,"delta":delta,"omega_idx":omega_idx,"error":str(e),"z_results":[]}

    g1=g//2
    im_diag_mean = float(np.mean(np.diag(Omega.imag)))
    offblock_norm = float(np.linalg.norm(Omega.imag[:g1,g1:])/np.sqrt(g1*(g-g1))) if delta>0 else 0.0
    im_eigs = np.linalg.eigvalsh(Omega.imag)

    z_results=[]
    for z_idx in range(N_Z):
        z_seed = Z_SEED_BASE + g*10000 + omega_idx*1000 + z_idx
        z = np.random.default_rng(z_seed).standard_normal(g)
        vn=theta_naive(z,Omega,N_CUT); vs=theta_s22(z,Omega,N_CUT)
        rel=float(abs(vn-vs)/abs(vn)) if abs(vn)>1e-300 else float("nan")
        log_rel=float(np.log10(rel)) if (not np.isnan(rel) and rel>0) else -16.0
        z_results.append({"z_idx":z_idx,"log10_rel_err":log_rel})

    lerrs=[r["log10_rel_err"] for r in z_results]
    return {"g":g,"delta":delta,"omega_idx":omega_idx,"omega_seed":seed,
            "im_diag_mean":im_diag_mean,"offblock_norm":offblock_norm,
            "im_min_eig":float(im_eigs.min()),"im_max_eig":float(im_eigs.max()),
            "median_log10":float(np.median(lerrs)),"mean_log10":float(np.mean(lerrs)),
            "std_log10":float(np.std(lerrs)),"min_log10":float(np.min(lerrs)),
            "max_log10":float(np.max(lerrs)),"z_results":z_results}

def compute_summary(records):
    summary={}
    all_g = sorted(set(r["g"] for r in records))
    for g in all_g:
        for delta in DELTAS:
            key=f"g{g}_d{delta:.1f}"
            sub=[r for r in records if r["g"]==g and abs(r["delta"]-delta)<1e-9 and not r.get("error")]
            if not sub: continue
            omega_medians=[r["median_log10"] for r in sub]
            omega_stds=[r["std_log10"] for r in sub]
            all_lerrs=[zr["log10_rel_err"] for r in sub for zr in r["z_results"]]
            summary[key]={"g":g,"delta":delta,"n_omega":len(sub),
                "global_median":float(np.median(all_lerrs)),
                "global_mean":float(np.mean(all_lerrs)),
                "global_std":float(np.std(all_lerrs)),
                "within_omega_std":float(np.mean(omega_stds)),
                "between_omega_std":float(np.std(omega_medians)),
                "omega_median_range":float(np.max(omega_medians)-np.min(omega_medians))}
    return summary

def save_results(records, summary, meta):
    with open(RESULT_PATH,"w",encoding="utf-8") as f:
        json.dump({"meta":meta,"summary":summary,"records":records},f,indent=2,ensure_ascii=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=max(1,os.cpu_count() or 1))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()

    setup_logging()
    t_start=time.perf_counter()

    all_tasks=[(g,delta,di,oi)
               for g,(oi_start,oi_end) in G_CONFIGS.items()
               for di,delta in enumerate(DELTAS)
               for oi in range(oi_start,oi_end)]
    total=len(all_tasks)

    log.info("="*65)
    log.info("omega_fixed_scan2.py  g=10異常追加調査")
    log.info(f"workers={args.workers}  総Omegaケース={total}  総theta計算={total*N_Z:,}")
    for g,(s,e) in G_CONFIGS.items():
        log.info(f"  g={g}: omega_idx {s}〜{e-1} ({e-s}個)")
    log.info("="*65)

    done_keys=set()
    done_records=[]
    if args.resume and Path(RESULT_PATH).exists():
        with open(RESULT_PATH) as f: done_records=json.load(f).get("records",[])
        for r in done_records: done_keys.add((r["g"],round(r["delta"],1),r["omega_idx"]))
        log.info(f"resume: {len(done_keys)}件スキップ")

    pending=[t for t in all_tasks if (t[0],round(t[1],1),t[3]) not in done_keys]
    log.info(f"実行対象: {len(pending)} / {total}")

    results=list(done_records); completed=len(done_records)

    with multiprocessing.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(run_omega_case, pending,
                                      chunksize=max(1,len(pending)//(args.workers*4))):
            results.append(r); completed+=1
            if r.get("error"):
                log.warning(f"ERROR {r}")
            elif completed%10==0 or completed==total:
                pct=100*completed/total
                log.info(f"[{completed:3d}/{total}] ({pct:.1f}%) "
                         f"g={r['g']} δ={r['delta']:.1f} oi={r['omega_idx']:2d} "
                         f"median={r.get('median_log10',float('nan')):.2f} "
                         f"std={r.get('std_log10',float('nan')):.2f}")
            if completed%30==0:
                save_results(results,compute_summary(results),
                    {"status":"running","completed":completed,
                     "timestamp":datetime.now().isoformat(timespec="seconds")})
                log.info(f"  → 中間保存 {completed}件")

    elapsed=time.perf_counter()-t_start
    summary=compute_summary(results)
    save_results(results,summary,{"status":"completed","completed":completed,
        "elapsed_s":round(elapsed,1),"elapsed_min":round(elapsed/60,2),
        "g_configs":G_CONFIGS,"N_cut":N_CUT,"deltas":DELTAS,"N_z":N_Z,
        "timestamp":datetime.now().isoformat(timespec="seconds")})

    log.info("="*65)
    log.info(f"完了: {completed}件  {elapsed/60:.1f}分")
    log.info("")
    log.info(f"{'g':>4} {'delta':>6} {'global_med':>11} {'within':>8} {'between':>9} {'range':>8}")
    log.info("-"*55)
    all_g=sorted(set(r["g"] for r in results))
    for g in all_g:
        for delta in DELTAS:
            s=summary.get(f"g{g}_d{delta:.1f}")
            if not s: continue
            log.info(f"{g:>4} {delta:>6.1f} {s['global_median']:>11.2f} "
                     f"{s['within_omega_std']:>8.2f} {s['between_omega_std']:>9.2f} "
                     f"{s['omega_median_range']:>8.2f}")
        log.info("")

if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
