# ニューラル変分スカーミオン — コードと run 記録

論文

> K. Hashimoto, M. Murata and G. Ogiwara,
> *Neural variational Skyrmions as a topological benchmark: emergent
> symmetry, precision energies and KAN–MLP comparison*,
> arXiv:2611.27411 [hep-th]（2026）、https://arxiv.org/abs/2611.27411

に付属する配布物です。ソースコード、設定ファイル、
論文が報告する全 run の記録（失敗した run を含む）が入っています。

英語版は `README.md`。

## 1. 中身

| パス | 中身 |
|---|---|
| `code/src/` | 模型そのもの。`physics.py`（エネルギー、バリオン密度、求積）、`train.py`（学習ループ、損失、格子のランダム化）、`models.py`（MLP、EffKAN、FastKAN、Fourier 特徴 MLP、pykan ラッパ） |
| `code/src/history/` | 一部の run が使った旧版の `physics.py` と `train.py` |
| `code/experiments/` | 実験ごとの実行スクリプト。その実験の設定値を保持 |
| `code/analysis/` | 記録から論文の表と図を作るスクリプト |
| `data/runs_reported.csv` | 論文が報告する 1,398 run を1行ずつ |
| `data/run_records/` | 同じ run の完全な記録（JSON） |
| `data/derived/` | 一部の表・図が読む中間結果 |
| `MANIFEST.md` | どのスクリプトが、どのファイルから、どの図表を作るか。環境も記載 |

別ファイルのアーカイブが2つあります（どのスクリプトに要るかは `MANIFEST.md`）。

* **アーカイブ1**（`skyrmion_histories.zip`）— エポックごとの $E$ と $B$ の履歴（1,398 run すべて）。図4、図5、表7 の電荷履歴に必要。
* **アーカイブ2**（`skyrmion_models.zip`）— 保存モデル（323 run）。スクリプトが読む 263 本と、論文の4節が述べる表1 の再実行 60 本です。表3、表9、図3 のデータ、付録A.4 のモード(c) の解析を保存モデルから計算し直すスクリプトに必要。計算の結果は `data/derived/` に入っているので、表と図3 はアーカイブ2 が無くても作れます。

この配布物（`skyrmion_release.zip`）と2つのアーカイブは、Zenodo（https://doi.org/10.5281/zenodo.24390286）にあります。
配布物の本体は GitHub（https://github.com/MasakiMurata1982/neural-variational-skyrmions）でも見られます。

どちらも `data/run_records/<experiment>/<runs_dir>/` に展開します。

## 2. 入っている run

1,398 本は**論文が報告する run のすべて**です。論文の表12 は、それを実験ごとに1行で並べています（20行）。
`runs_reported.csv` の `cohort_inventory` 列が行を示し、
`make_inventory_from_reported.py` が表12 の行の順に数えます。

```
§4
600   sweep           anchored 掃引（図2、表1、付録A.2・A.5）
 60   t1_lr           表1 の2行目：学習率 5e-4、1e-3、5e-3、1e-2、3ネットワーク、5シード
 30   t1_time         表1 の3・4行目：30 s と 120 s で止めた run、3ネットワーク、5シード
 10   t1_seeds59      KAN[32]×3 のシード5〜9、学習率 5e-3 と 1e-2（§4 の学習率の脚注）
 60   t1_regrid       表1 の60本の再実行、80×44×60 格子でも評価
§5
240   uniform         統一プロトコル：KAN[16]×4、KAN[21]×4、GELU-MLP[64]×4、GELU-MLP[48]×4（表2）
 80   wallclock       時間を揃えた比較：KAN[16]×4 を 160 s、GELU-MLP[64]×4 を 660 s（表6）
§6
 16   t7              表7 の16本
  1   fig5_imprinted  図5 の刷り込みありの run
 50   design          刷り込みの回帰をどこまで進めるかを変えた run（表8）
§7
 12   kan_grid        KAN[16]×4 と KAN[8]×2 の G = 3 と G = 10（§7.3）
 13   repro13         GPU の非決定性を測る再実行13本（§7.4）
付録A
144   fastkan         anchored トラックの FastKAN（付録A.2・A.5）
  2   a4_vzero        真空にほどけた2本の再実行（付録A.4）
  5   a5_regrid       B = 1 の anchored run 5本の再実行、80×44×60 格子でも評価（付録A.5）
付録B
 36   b1_lr           付録B.1 (i)：KAN[16]×4 と [32]×4 の両 MLP、学習率4つ、3シード
  9   b1_silu         付録B.1 (ii)：SiLU の MLP、幅3つ、3シード
 12   b1_time         付録B.1 (iii)：KAN[16]×2 と tanh-MLP[32]×4 を 30 s と 120 s で止めた run
 15   pykan           anchored トラックの pykan（付録B.2）
付録D
  3   appD            表2 の3本の再実行、訓練中のバリオン密度を保存（図6）
```

付録D の3本は、繰り返した表2 の run と**同じ `run_id`** です。`record_path` で区別してください。
再実行は `2026-11-18_appD/runs/`、表2 の run は `2026-08-31_free_uniform/runs_uniform/` にあります。

## 3. 数値の再現

`MANIFEST.md` が図表ごとに「スクリプト → 入力 → 出力」を書いています。スクリプトは
CSV を第1引数で受け、既定は `data/runs_reported.csv` です。

```
cd code/analysis
python3 make_tables_from_csv.py          # 表2・4・6・10
python3 classify_failures.py             # 表5・11
python3 table1_recompute_v2.py           # 表1
python3 make_appendixB1_fairness.py      # 付録B.1 の3項目と §4 の学習率の脚注
python3 make_inventory_from_reported.py  # 表12
```

この5本は標準ライブラリだけで動き、**記録だけから表が出ます。**アーカイブ1・2 や
GPU が要るスクリプトは `MANIFEST.md` に印があります。

スクリプトが出すのは表と図です。論文の本文に書いてあるそのほかの数（p 値、範囲、本数など）は、
`data/runs_reported.csv` の列から計算できます。すべてにスクリプトがあるわけではありません。

run を走らせ直すには、`MANIFEST.md` の「Re-training」の節を見てください。表12 の行ごとに、
`code/experiments/` の実行スクリプトと引数を書いてあります。

図6 は `data/derived/appD_frames/` の場21枚から描きます（scikit-image が必要）。

```
python3 make_fig_relaxation.py ../../data/derived/appD_frames fig_relaxation.png
```

解析スクリプトは Python 3 と numpy・scipy・matplotlib、再学習には PyTorch と GPU が
必要です。動作確認した版は `MANIFEST.md` にあります。

## 4. 記録を使う前に知っておくこと

列の説明は `data/runs_reported_README.md` にあります。

### (1) pykan の 15 本は、3シードが同じ run の繰り返し

pykan 0.2.8 は `__init__` の中で `torch.manual_seed` を呼びます。これらの run を作ったとき、
`models.py` のラッパはシードを渡していなかったので、シードが効いていません。pykan の 15 本は
5 つの構成（電荷・幅・深さ）で、**構成ごとの3本は、同じ初期値・同じ求積列で同じ設定を3回
走らせたもの**です。3本のエネルギーは $2.2\times10^{-7}$ 以内で一致します。

論文はこの 15 本を独立なシードとして使っていません。**図2 は pykan の系列を描いておらず**、
付録B.2 が時間の比較に使うだけです。記録からシードの分散を計算する場合は、pykan の run を除いてください。

### (2) ソースが変わっている

* `physics.py` — 2026-09-07 に、1次元プロファイル前学習の**固定シード再スタート**を追加。
  **2026-09-08 に、再スタートを採用する閾値を $10^{-9}$ から $10^{-6}$ に上げました。**
  anchored 掃引の B=2、シード 3・4 の run は、直したプロファイルで走らせ直しました
  （`2026-09-11_refit_v2/runs_refit_v2`、96 本）。
* `train.py` — 2026-09-14 に、刷り込み回帰の**目標損失での早期停止**を追加。表8 の設計
  実験だけが使います。

* `models.py` — pykan のラッパが、シードを渡すようになりました（上の (1)）。

どの版がどの run を作ったかは `MANIFEST.md` の表にあります。`physics.py` と `train.py` の旧版は
`code/src/history/` に入れてあります。

### (3) 走らせ直しても、記録と bit 一致するとは限らない

論文の §7.4 がこの測定を報告しています。元のシードで走らせ直した 13 本のうち、9 本は記録と
bit 一致しました。残りの 4 本は、エネルギーが $2.1\times10^{-8}$（2 本）、$1.6\times10^{-5}$、
$1.5\times10^{-4}$（幅 8 のネットワーク）違いました。anchored トラックの実行スクリプトは、最初の run の
前に 1 次元プロファイルを 1 回学習します（`torch.manual_seed(0); get_base_profile(B, device)`）。
これを省くと、プロセスの最初の anchored の run が記録とずれます。

## 5. ライセンス

この配布物のコードとデータは CC BY 4.0 で提供します（`LICENSE` を参照）。利用の際は論文を引用してください。

EffKAN 層は efficient-kan（MIT）の設計に従っています。pykan（MIT）は付録B.2 の参照実装としてのみ使っています。
