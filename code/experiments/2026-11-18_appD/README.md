# 付録 D(図 6)の材料 — 2026-11-18

論文の付録 D と図 6(訓練中のバリオン密度の等値面)の元になった run の記録と、図を作るのに使う場。

## 何を走らせたか

- 表 2 の 3 本を、同じ設定・同じシードで走らせ直した。GELU-MLP[64]×4、20,000 エポック、B=2 シード 7、B=3 シード 13、B=4 シード 16。
  元の記録は `2026-08-31_free_uniform/runs_uniform/` の同名の run。
- 機械と環境:RTX 6000 Ada、PyTorch 2.10.0+cu128、numpy 2.2.6。記録を作ったときと同じ torch の版。
- コード:配布物の `physics.py`・`models.py` と `train_dump.py`。
  `train_dump.py` は配布物の `train.py` を写し、場を書き出す関数 1 つと、その呼び出し(最初のエポックの前と、指定したエポックの後)を足したもの。ループ・勾配・乱数の消費は変えていない。
- 実行は `run_all4.py`。プロセスの 1 本目のずれを避けるため、最初に B=1(GELU-MLP[48]×4 シード 4)を走らせる(その結果は使わない)。`run_all4.py` は、このフォルダの `dump_B1/`〜`dump_B4/` に記録と場を書く。

## 結果

3 本とも、各エポックの E と B が元の記録とビット単位で一致した(`cmp_hist.py`、元の履歴はアーカイブ 1)。最終の E は 2.356966040244 / 3.437086811959 / 4.480050170227。

## 中身

| 場所 | 中身 |
|---|---|
| `data/run_records/2026-11-18_appD/runs/` | 3 本の run の記録:`.json`(設定と検証の値)。`_hist.npz`(E と B の履歴)はアーカイブ 1 を展開するとここに入る |
| `data/derived/appD_frames/dump_B{2,3,4}/` | 図 6 が使う場 21 枚(各電荷 7 エポック)と、塊の中心を読むための run の `.json` |
| このフォルダ | `train_dump.py`・`run_all4.py`(再実行)、`cmp_hist.py`(記録との比較)。図 6 を描く `make_fig_relaxation.py` は `code/analysis/` にある |

- 場のファイル `dump_epNNNNN.npz` のキー:`vol`(バリオン密度、80×80×80)、`lin`(各軸の座標、−4 から 4)、`epoch`。
- 再実行では 1 本あたり 110 枚を書き出した(3 本で約 600 MB)。ここには図 6 が使う 21 枚だけを入れている。残りは `train_dump.py` で作り直せる。
- 図 6 のエポック:B=2 は 0, 52, 68, 84, 100, 164, 20000。B=3 は 0, 100, 340, 400, 460, 580, 20000。B=4 は 0, 100, 600, 1000, 1200, 1500, 20000。

## 図 6 の作り方

```
cd code/analysis
python3 make_fig_relaxation.py ../../data/derived/appD_frames fig_relaxation.png
```

matplotlib 3.11.2(scikit-image 0.26.0)で、論文の `fig_relaxation.png` と byte 単位で一致する(上の再実行の環境とは別の版。ほかの版では図の大きさが 1 行変わることがあるので、図を見比べる)。

## 注意

3 本の `run_id` は、表 2 の元の run と同じ名前。区別は記録の場所(`record_path`)で行う。`runs_reported.csv` では `cohort_inventory` が `appD` の 3 行。
