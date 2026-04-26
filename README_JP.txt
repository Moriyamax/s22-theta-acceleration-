S(2,2) 分解による高次元リーマン Theta 関数の高速計算
GL(4) Hitchin 系ベンチマーク（s22 + naive のみ）

このリポジトリは、GL(4) Hitchin 系に現れる高次元リーマン Theta 関数を
2つのモードで計算し、計算時間を比較するためのベンチマーク実装です。

精度と正確性について
本手法(因子分解)はtheta関数の恒等式に基づいているため、
**周期行列が完全にブロック対角である条件下**では、理論上の誤差はゼロです。
生じ得る誤差は、標準的な浮動小数点演算に固有の丸め誤差のみです。

s22   : S(2,2) 分解を用いて g を (g1, g2) に分割し、2つの naive 計算を行う
naive : g 次元の Theta 関数をそのまま打ち切り級数で計算する

S(2,2) 上では周期行列がほぼブロック対角になるため、
g=17 の Theta 関数を 8 次元 + 9 次元に分割して計算できます。
その結果、計算時間は大幅に短縮されます。

例（g=17, N_cut=2）:
naive（17次元）推定: 約 3,900,000 秒（約 45 日）
s22（8+9 次元）実測: 13.1 秒

約 10^5 倍の高速化になります。


--

1.特徴

s22 と naive の 2 モードのみを実装

自動レジューム機能（results.json を参照して未完了ケースのみ実行）
進捗ログ（progress_log.txt）に ETA と完了予測時刻を記録
g の上限をモードごとに設定可能（s22_g_max, naive_g_max）
実行結果は JSON 形式で保存（値の実部・虚部、環境情報つき）

2.背景
g 次元のリーマン Theta 関数は
(2 * N_cut + 1)^g
個の項を持ちます。

g=17, N_cut=2 の場合:
5^17 = 7.6e11 項
となり、naive計算は現実的ではありません。

一方、S(2,2) 上ではスペクトル曲線が可約になり、
Ctilde = Ctilde_1 ∪ Ctilde_2
周期行列 Omega がほぼブロック対角になります:
Omega ≈ diag(Omega_1, Omega_2)

このとき、
theta(z | Omega) ≈ theta(z1 | Omega_1) * theta(z2 | Omega_2)
となり、17 次元の計算を 8 次元 + 9 次元に分割できます。

3.ファイル構成
ScanS22.py
メインのベンチマークスクリプト。
s22 / naive の 2 モード、レジューム、ETA、サマリー出力を実装。

config.txt
初回実行時に自動生成される設定ファイル。

progress_log.txt
実行中の進捗ログ（%・ETA・完了予測時刻）。

results.json
各ケースの結果を保存。
- mode
- g
- N_cut
- time_s
- n_terms
- value_re, value_im
- 実行環境情報
- timestamp

4.実行方法
python ScanS22.py

スクリプトは以下を行います:
config.txt を読み込む
resume=true の場合、results.json を参照して未完了ケースのみ実行
各 (mode, g, N_cut) の計算を実行
進捗と ETA を progress_log.txt に記録
結果を results.json に逐次保存
最後にサマリーを表示

5.設定ファイル（config.txt）

例:
g_list = 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
N_cut_list = 1,2
mode = all
seed = 42
report_every = 1000000
resume = true
s22_g_max = 0
naive_g_max = 0

説明:
mode
s22   : s22 のみ
naive : naive のみ
all   : s22 → naive の順に実行

s22_g_max, naive_g_max
正の整数を指定すると、その g を超えるケースをスキップ
0 または省略で制限なし

6.出力サマリー
実行終了後、以下を表示します:
モード別の結果一覧（g, N_cut, 項数, 時間）
naive/s22 の削減比
naive g=17 の外挿推定（未実測の場合）
naive/s22 の速度比

7.注意事項
理論上の誤差がゼロになるのは”周期行列が完全にブロック対角”である条件下のみです。

8.ライセンス
MIT License

