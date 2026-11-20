# runs_reported.csv の列の説明

`runs_reported.csv` は、論文が報告する 1,398 本の run を 1 行ずつ並べた表です。値は run の記録(`data/run_records/` の JSON)を読んだもので、
**再学習・再評価は行っていません**。記録に無い値は空欄です。
列 `cohort_inventory` は論文の表 12 の行を表します(値の一覧は README の 2 節)。ほかの列を下に説明します。セクターを保っているかどうかは、`B` と `B_target` の差(|B − B_target| < 10⁻³)で判定します。

## 1. 列の定義

「hard」と「anchored」は同じもの(コード上の名前が hard、論文上の名前が anchored)。

### 識別
| 列 | 内容 |
|---|---|
| `run_id` | レコードの tag。**フォルダをまたぐと一意ではない**(例: 付録 D の再実行 3 本と付録 A.4 の再実行 2 本は、表 2 の元の run と同じ tag)。一意キーは `record_path`。 |
| `record_path` | `data/run_records/` からの相対パス(`<tag>.json`) |
| `experiment`, `runs_dir` | 実験フォルダ名、その中の runs サブディレクトリ名 |
| `date` | 実験フォルダ名の日付(実験開始日)。レコードにタイムスタンプは無い |
| `cohort` | 論文の表 12 の行の説明(`cohort_inventory` が同じ run は同じ文) |

### 分類
| 列 | 内容 |
|---|---|
| `track` | `hard`(anchored)/ `free` / `soft`(表 7 の群 1・群 2。下の 4 節の 5) |
| `arch` | `effkan` / `mlp_gelu` / `mlp_tanh` / `mlp_silu` / `pykan` / `fastkan` |
| `kind` | レコード上のモデル種別(`mlp` は活性化を区別しない) |
| `width`, `depth` | 隠れ層の幅と層数(`[width]×depth`) |
| `n_params` | 学習可能パラメータ数 |
| `kan_grid` | KAN 系のみ。B スプライン(effkan/pykan)または RBF(fastkan)の grid 数。MLP は空欄 |

### 設定
| 列 | 内容 |
|---|---|
| `B_target` | 目標電荷 n |
| `seed` | 乱数シード |
| `epochs_config` | 設定上のエポック数(壁時計予算 run では**上限**) |
| `epochs_run` | 実際に回ったエポック数(= 履歴 npz の長さ)。壁時計予算 run では実測速度に依存する |
| `wallclock_budget_s` | 壁時計予算 [s]。無ければ空欄(= 予算なし) |
| `optimizer`, `lr`, `schedule` | すべて Adam + コサイン減衰(終値 0.02 lr)。予算 run は最初の 10 エポックの実測速度から T を見積もる |
| `w_topo` | 電荷ペナルティ重み(hard は 200、free 統一プロトコルは 1000 など) |
| `w_far` | 遠方真空項の重み(free のみ意味を持つ。hard/soft では 0 か空欄) |
| `r_max_topo` | 電荷ペナルティを打ち切り球で測った run の半径(注入実験)。通常は空欄 |
| `grid_train` | 学習求積格子(n_r × n_θ × n_φ、毎ステップランダム回転+動径写像 a を 5 値から無作為) |
| `w_origin` | soft/free のみ: 原点アンカー U(0) = −1 の罰則重み(表 7 の群 2〜4 は 10.0、他は 0。キーの無い記録は空欄 = 0)。 |
| `soft_scale` | soft のみ: q = normalize(q_vac + g_soft·soft_scale·NN) の係数(全 run 3.0)。 |
| `lump_spread` | 刷り込みをした free run のみ: 塊の中心座標のガウス標準偏差(配布物の run はすべて 1.5)。 |
| `lump_max_r` | 同上: 中心座標のクランプ半径(統一プロトコル・設計実験は 2.5)。**空欄 = 当時のコードにこの設定が無く、クランプなしで走った run**(`2026-08-28_B3_ansatz_free/runs_seeded2` の 1 本)。現行コードで再実行すると既定 2.5 のクランプが掛かり塊配置が変わり得る。 |
| `pretrain_epochs`, `pretrain_lr`, `n_lumps` | 刷り込み(free のみ)。刷り込みなしの free/hard/soft は空欄 |

### 結果
| 列 | 内容・単位 |
|---|---|
| `E` | 最終エネルギー [12π² 単位]。`grid_eval` の格子での値 |
| `grid_eval` | `E`, `B`, `E2`, `E4`, `delta_E` を評価した格子。JSON レコードは全 run で **48×28×36、動径写像 a = 2.6/3.0/3.4 の 3 格子(各 1 回ランダム回転)の平均**。 |
| `B` | 測定電荷(同じ評価) |
| `E2`, `E4`, `E2_over_E4` | 2 次項・Skyrme 項のエネルギーとビリアル比 |
| `delta_E` | 3 評価格子間の E の最大差 ΔE(求積悪用の検査) |
| `imprint_loss` | 刷り込み回帰の最終 MSE(free で刷り込みをした run のみ) |
| `imprint_loss_eval` | 刷り込み終了時の損失を固定の大バッチ(100 × 2048 点、seed 12345、全 run 共通)で測り直した値(2026-09-14 以降の run のみ。imprinting 設計実験の横軸) |
| `pretrain_target_loss` | 刷り込みの目標損失(指定した run のみ。直近 50 ステップ平均がこれを下回ったら早期停止、`pretrain_epochs` は上限) |
| `pretrain_epochs_used` | 実際に回った刷り込みステップ数(2026-09-14 以降の run) |
| `pretrain_reached` | 目標損失に到達したか(目標を指定した run のみ) |
| `wallclock_s` | 学習時間の実測 [s](刷り込み込み。評価時間は含まない) |
| `peak_mem_GiB` | GPU ピークメモリ [GiB] |
| `E_80x44x60`, `B_80x44x60`, `delta_E_80x44x60` | **最細格子 80×44×60**(a = 2.6/3.0/3.4、各 1 回転)での再評価値。統一プロトコル系の成功 run(`highres_all.json` 217 本 + KAN[21] B=1,2 の 20 本)、runs_b1_regrid 5 本(保存モデル再評価)、表 1 再実行 60 本。他は空欄(= 再評価していない) |

### 判定(論文 3.5 節のプロトコル、個別列)
| 列 | 定義 |
|---|---|
| `pass_virial` | \|E2/E4 − 1\| < 0.05 |
| `pass_dE` | ΔE < 10⁻² |
| `success_uniform` | free のみ: `pass_virial` かつ `pass_dE` かつ \|B − B_target\| < 10⁻³ かつ \|E − E_ref\|/E_ref < 1%(E_ref = 1.2322 / 2.3582 / 3.4386 / 4.4804)。統一プロトコル以外の free run にも同じ式で付けている(参考値)。hard/soft は空欄 |
| `uniform_protocol` | 統一プロトコル系(`runs_uniform*`、kan21、tb、mlp_tb、repro)の run か |
| `verification_rerun` | True/False。既存 run と同シード・同設定の検証用再実行(runs_b1_regrid 5、2026-09-11_table1_models 60、2026-09-11_refit_v2/runs_determinism_rep1,2 10)。シードの統計からは除く(`verification_rerun != "True"`)。ほかの再実行(`runs_uniform_repro` の 3 本、`2026-11-18_appD` の 3 本、`2026-10-01_vzero` の 2 本)は False のままなので、`runs_dir` で除く(解析スクリプトはそうしている)。run の記録(JSON)の `rel_err_vs_star` は `physics.py` の `E_STAR` に対する相対誤差(%)で、論文は使わない |

### 由来
| 列 | 内容 |
|---|---|
| `used_in` | 論文のどこで使っているか(節・図・表・付録の番号)。論文の表 12 の行による対応で、`cohort_inventory` が同じ run はすべて同じ値。個々の run が実際に図表の数値に寄与したかは別(例: 表 2 は成功・失敗を問わず全 run) |
| `has_hist`, `has_model` | 履歴(`_hist.npz`、アーカイブ 1。1,398 本すべてにあるので全行 True)/ 学習済みモデル(`_model.pt`、アーカイブ 2 の 323 本。解析スクリプトが読む 263 本と、論文の 4 節が述べる表 1 の再実行 60 本)が配布物にあるか |
| `notes` | 自動付与の特記(未収束、壁時計予算、FastKAN、soft、再学習、失敗モード分類など) |

## 2. 空欄の意味

- 「その track・設定に存在しない」: `kan_grid`(MLP)、`pretrain_*`/`imprint_loss`/`n_lumps`(刷り込みなし)、
  `wallclock_budget_s`(予算なし)、`r_max_topo`(打ち切りなし)、`success_uniform`(hard/soft)。
- 「記録に無い」: `E_80x44x60` 系(再評価していない run)。

## 3. 格子の表記と評価条件

- 表記は `n_r x n_theta x n_phi`。動径は r = a s/(1−s) の半無限写像 + Gauss–Legendre、r_max = 60。
- 学習: 32×20×24、毎ステップ SO(3) ランダム回転、a ∈ {2.4, 2.7, 3.0, 3.3, 3.6} を無作為(全 run 共通)。
- 最終評価(JSON レコードの `E` など): 48×28×36、a ∈ {2.6, 3.0, 3.4} の 3 格子・各 1 回転(乱数 seed 1234)の平均、ΔE は 3 格子の最大差(全 run 共通)。
- 最細格子 80×44×60(同じ 3 写像・回転): `E_80x44x60` 列に値のある run のみ。
  統一プロトコル系の成功 run(237 本)で、80 格子と 48 格子の E の差は、MLP(124 本)で −6×10⁻⁷ から +5×10⁻⁶、KAN(113 本)で −2.9×10⁻⁴ から +1.3×10⁻⁴。論文の代表値(表 4・付録 A.1)は 80 格子の値。

## 4. 記録の作りに由来する注意点

1. **anchored(hard)の run はモデルを保存していない**(`has_model` = False)。再評価には同シード再学習が必要
   (保存モデルつきの再実行は `verification_rerun` = True の run)。
2. **壁時計予算の run**(`wallclock_budget_s` あり)は `epochs_run` が実測速度に依存する。例: GELU-MLP[64] 120 s は 4,973〜5,331 ep、
   KAN[16] 160 s は約 4,850 ep、GELU-MLP[64] 660 s は約 8 万 ep。`epochs_config` は上限。
3. **1 次元プロファイル(anchored の背景場)の学習はシード設定の順序に依存する。** `train_one` は `manual_seed(seed)` の後に
   プロファイルを学習するので全レコードは E₀ = 2.416270 の正しい解を使っているが、シード未設定で先に学習すると L-BFGS が
   E₀ = 2.416352 の劣った解に落ち、その場合 80 格子の E が +1.5×10⁻⁴、B が 2.00005 にずれる。
   レコードから再現する際はこの順序を守ること。
4. **同一 tag の重複**: `run_id` は tag なので、再実行の run は元の run と同 tag(99 個の tag が 2 回以上現れる)。一意キーは `record_path`。
5. **soft トラック(8 本)**: 真空土台 + 減衰窓の構成で、電荷が B = 0 に凍結して目標セクターへ出られなかった構成。
   E ≈ 0 の run では virial が定義できず `pass_virial` は False になる。論文では表 7 の群 1・群 2。
6. **`2026-08-28_B3_ansatz_free/runs_free2`** は電荷ペナルティを打ち切り球 r ≤ r_max_topo で測る注入実験(`r_max_topo` 列)。B が量子化されない設計である。
7. **時期による config 項目の差**: 初期のフォルダの記録には、後から追加された設定キー
   (`mlp_act`, `w_far`, `time_budget_sec` …)が無い。その場合は当時の既定値(mlp_act = tanh、w_far = 0、予算なし)に相当し、CSV では空欄または既定値で出している(`w_far` は空欄)。
