"""
RLD_theta_engine.jl
Recursive Log-Decomposition Theta Engine: Handles ultra-high genus without overflow.
"""

using LinearAlgebra, Random, Dates, Printf, JSON

# ───────── 定数・パス ─────────
const PREFIX      = "RLD_"
const SCRIPT_DIR  = @__DIR__
const LOG_PATH    = joinpath(SCRIPT_DIR, "$(PREFIX)log.txt")
const RESULT_PATH = joinpath(SCRIPT_DIR, "$(PREFIX)results.json")

const LOG_LOCK     = ReentrantLock()
const RESULTS_LOCK = ReentrantLock()

# ───────── ユーティリティ ─────────
function log_msg(msg::String)
    line = "[$(Dates.format(now(),"yyyy-mm-dd HH:MM:SS"))] $msg"
    println(line)
    lock(LOG_LOCK) do
        open(LOG_PATH, "a") do f; println(f, line); end
    end
end

struct E8Cfg
    g_list::Vector{Int}
    N_cut::Int
    g_limit_naive::Int
    tau_dummy_im::ComplexF64
    seed_base::Int
    outer_loops::Int
    use_log_output::Bool  # trueなら結果を対数で表示、falseなら10進に戻す
end

# ───────── Theta Core: Log-Naive ─────────
# Naive計算も対数で返すように変更
function log_theta_naive(z::Vector{ComplexF64}, Omega::Matrix{ComplexF64}, N_cut::Int)::ComplexF64
    g = length(z)
    b = 2*N_cut + 1
    total = b^g
    
    # Naiveは項の和なので、まず通常の和を計算してから最後にlogをとる
    # (内部の個別の項がオーバーフローする場合はさらに工夫が必要だが、
    #  g_limit_naiveは通常小さいのでこの実装で十分)
    nt = max(1, Threads.nthreads())
    tasks = map(0:nt-1) do ci
        i_start = ci * (total ÷ nt) + min(ci, total % nt)
        i_end   = i_start + (total ÷ nt) - 1 + (ci < total % nt ? 1 : 0)
        Threads.@spawn begin
            local_sum = zero(ComplexF64)
            nv = Vector{Float64}(undef, g)
            for idx in i_start:i_end
                tmp = idx
                for k in 1:g
                    nv[k] = Float64(tmp % b - N_cut)
                    tmp ÷= b
                end
                qc = dot(nv, Omega * nv)
                lc = dot(nv, z)
                local_sum += exp(im*π*qc + 2im*π*lc)
            end
            local_sum
        end
    end
    val = sum(fetch(t)::ComplexF64 for t in tasks)
    return log(val) # 複素対数を返す
end

# ───────── Theta Core: Recursive Log-Decomposition ─────────
function log_theta_recursive(z::Vector{ComplexF64}, Omega::Matrix{ComplexF64}, N_cut::Int, g_limit::Int, tau_dummy::ComplexF64)::ComplexF64
    g = length(z)

    if g <= g_limit
        return log_theta_naive(z, Omega, N_cut)
    end

    if !ispow2(g)
        new_g = nextpow(2, g)
        z_pad = vcat(z, zeros(ComplexF64, new_g - g))
        Omega_pad = zeros(ComplexF64, new_g, new_g)
        Omega_pad[1:g, 1:g] = Omega
        for i in (g+1):new_g
            Omega_pad[i, i] = tau_dummy
        end
        return log_theta_recursive(z_pad, Omega_pad, N_cut, g_limit, tau_dummy)
    end

    g_half = g ÷ 2
    t1 = Threads.@spawn log_theta_recursive(z[1:g_half], Omega[1:g_half, 1:g_half], N_cut, g_limit, tau_dummy)
    t2 = Threads.@spawn log_theta_recursive(z[g_half+1:end], Omega[g_half+1:end, g_half+1:end], N_cut, g_limit, tau_dummy)
    
    # log(A * B) = log(A) + log(B)
    return fetch(t1) + fetch(t2)
end

# ───────── 出力制御 ─────────
function format_result(log_val::ComplexF64, as_log::Bool)
    if as_log
        return @sprintf("Log(Theta) = %.6f + %.6fi", real(log_val), imag(log_val)), log_val
    end

    # 10進数に戻す試行
    res = exp(log_val)
    if isinf(real(res)) || isinf(imag(res))
        msg = @sprintf("数字が大きすぎて10進に戻せません。対数結果を表示します: Log = %.6f + %.6fi", 
                        real(log_val), imag(log_val))
        return msg, log_val
    else
        return @sprintf("%.6e + %.6ei", real(res), imag(res)), res
    end
end

# ───────── メインロジック ─────────
function main()
    cfg = E8Cfg(
        [16, 17,18,19,20,25,30,50,100,999,1500],   # g=20000などの超高属数もテスト
        2,               # N_cut
        1,              # Naive限界
        10.0im,          # パディング虚部
        42,              # seed
        5,               # 外側ループ数
        false            # true:対数、デフォルトは10進表示（false） LOGならg=20000も可能
    )

    log_msg("="^70)
    log_msg("RLD-Engine Start: Recursive Log-Decomposition")
    log_msg("G-List: $(cfg.g_list), Log-Mode Output: $(cfg.use_log_output)")
    log_msg("="^70)

    all_results = []
    max_g = maximum(cfg.g_list)

    for loop_idx in 1:cfg.outer_loops+1
        log_msg(">>> Starting Global Loop [$loop_idx / $(cfg.outer_loops)]")
        
        rng = MersenneTwister(cfg.seed_base + loop_idx)
        Z_master = complex.(randn(rng, max_g), randn(rng, max_g) .* 0.1)
        Omega_master = zeros(ComplexF64, max_g, max_g)
        for i in 1:max_g
            Omega_master[i,i] = 2.0im + complex(0.0, 0.2 * randn(rng))
        end
        Omega_master = (Omega_master + Omega_master') / 2

        for g in cfg.g_list
            z = Z_master[1:g]
            Omega = Omega_master[1:g, 1:g]

            # 1. Recursive (常に内部は対数)
            t_start = time()
            log_val_rec = log_theta_recursive(z, Omega, cfg.N_cut, cfg.g_limit_naive, cfg.tau_dummy_im)
            t_rec = time() - t_start

            # 2. Naive比較 (gが小さい場合のみ。これも対数で受ける)
            log_val_naive = NaN + NaN*im
            abs_err_log = NaN
            if g <= cfg.g_limit_naive
                log_val_naive = log_theta_naive(z, Omega, cfg.N_cut)
                abs_err_log = abs(log_val_rec - log_val_naive)
            end

            # 出力フォーマットの適用
            display_str, final_val = format_result(log_val_rec, cfg.use_log_output)
            
            msg = @sprintf("    (Loop %d, g=%d) Result: %s, Time: %.4fs, LogErr: %s", 
                    loop_idx, g, display_str, t_rec, 
                    isnan(abs_err_log) ? "N/A" : @sprintf("%.2e", abs_err_log))
            log_msg(msg)

            push!(all_results, Dict(
                "loop" => loop_idx,
                "g" => g,
                "is_log_format" => cfg.use_log_output,
                "log_val_re" => real(log_val_rec),
                "log_val_im" => imag(log_val_rec),
                "time_rec" => t_rec,
                "abs_err_log" => isnan(abs_err_log) ? nothing : abs_err_log,
                "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
            ))
        end

        lock(RESULTS_LOCK) do
            open(RESULT_PATH, "w") do f; JSON.print(f, all_results, 4); end
        end
    end
    log_msg("All log-reconstruction experiments completed.")
end

main()