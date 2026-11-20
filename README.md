# Neural variational Skyrmions — code and run records

Companion archive to

> K. Hashimoto, M. Murata and G. Ogiwara,
> *Neural variational Skyrmions as a topological benchmark: emergent
> symmetry, precision energies and KAN–MLP comparison*,
> preprint (2026).

This archive holds the source code, the configuration files and the records
of every run the paper reports, including every failed run.

## 1. What is here

| path | contents |
|---|---|
| `code/src/` | the model. `physics.py` (energy, baryon density, quadrature), `train.py` (training loop, loss, grid randomisation), `models.py` (MLP, EffKAN, FastKAN, Fourier-feature MLP, pykan wrapper) |
| `code/src/history/` | the two earlier versions of `physics.py` and `train.py` that some runs used |
| `code/experiments/` | one driver script per experiment, holding the settings of that experiment |
| `code/analysis/` | the scripts that turn the records into the tables and figures of the paper |
| `data/runs_reported.csv` | one row per run, for the 1,398 runs the paper reports |
| `data/run_records/` | the full record of each of those runs, as JSON |
| `data/derived/` | the intermediate results that some tables and figures read |
| `MANIFEST.md` | which script makes which table or figure, from which file, and in which environment |

Two further archives are distributed as separate files (the scripts that need them are listed in `MANIFEST.md`):

* **archive 1** (`skyrmion_histories.zip`), the per-epoch histories of energy and charge of all 1,398 runs. Figure 4,
  figure 5 and the charge histories of table 7 need it.
* **archive 2** (`skyrmion_models.zip`), the saved models (323 runs): the 263 that the scripts read, and the 60 re-runs of table 1 that
  section 4 of the paper describes. The scripts that compute table 3, table 9, the
  data of figure 3 and the mode-(c) analysis of appendix A.4 from the models need it. Their
  results are stored in `data/derived/`, so the tables and figure 3 can be made without it.

This package (`skyrmion_release.zip`) and the two archives are deposited at Zenodo, https://doi.org/10.5281/zenodo.24390286.
The package is also at https://github.com/MasakiMurata1982/neural-variational-skyrmions.

Both unpack into `data/run_records/<experiment>/<runs_dir>/`.

## 2. Which runs are here

The 1,398 runs are every run the paper reports. Table 12 of the paper lists them,
one row per experiment: 20 rows. `runs_reported.csv` carries a column
`cohort_inventory` that names the row each run belongs to, and
`make_inventory_from_reported.py` counts the rows
in the order of table 12:

```
Section 4
600   sweep           anchored sweep (figure 2, table 1, appendices A.2 and A.5)
 60   t1_lr           table 1, row 2: learning rates 5e-4, 1e-3, 5e-3 and 1e-2, three networks, five seeds
 30   t1_time         table 1, rows 3 and 4: stopped after 30 s and after 120 s, three networks, five seeds
 10   t1_seeds59      seeds 5 to 9 of KAN[32]x3 at learning rates 5e-3 and 1e-2 (footnote of section 4 on the best learning rate)
 60   t1_regrid       re-runs of the 60 runs behind table 1, evaluated also on the 80x44x60 grid
Section 5
240   uniform         uniform protocol: KAN[16]x4, KAN[21]x4, GELU-MLP[64]x4, GELU-MLP[48]x4 (table 2)
 80   wallclock       equal wall-clock budget: KAN[16]x4 for 160 s, GELU-MLP[64]x4 for 660 s (table 6)
Section 6
 16   t7              the sixteen runs of table 7
  1   fig5_imprinted  the imprinted run of figure 5
 50   design          runs that vary how far the imprinting regression is taken (table 8)
Section 7
 12   kan_grid        KAN[16]x4 and KAN[8]x2 at G = 3 and G = 10 (section 7.3)
 13   repro13         thirteen re-runs to measure GPU nondeterminism (section 7.4)
Appendix A
144   fastkan         anchored-track FastKAN runs (appendices A.2 and A.5)
  2   a4_vzero        re-runs of the two runs that unwind to the vacuum (appendix A.4)
  5   a5_regrid       re-runs of five anchored runs at B = 1, evaluated also on the 80x44x60 grid (appendix A.5)
Appendix B
 36   b1_lr           appendix B.1 (i): KAN[16]x4 and both MLPs at [32]x4, four learning rates, three seeds
  9   b1_silu         appendix B.1 (ii): MLP with SiLU activation, three widths, three seeds
 12   b1_time         appendix B.1 (iii): KAN[16]x2 and tanh-MLP[32]x4 stopped after 30 s and after 120 s
 15   pykan           anchored-track pykan runs (appendix B.2)
Appendix D
  3   appD            re-runs of three runs of table 2, saving the baryon density during training (figure 6)
```

The three runs of appendix D have the same `run_id` as the runs of table 2 that
they repeat. Tell them apart by `record_path`: the re-runs are in
`2026-11-18_appD/runs/`, the runs of table 2 in `2026-08-31_free_uniform/runs_uniform/`.

## 3. How to reproduce a number

`MANIFEST.md` maps every figure and table to its script and its input. The
scripts take the CSV as an optional first argument and default to
`data/runs_reported.csv`, so

```
cd code/analysis
python3 make_tables_from_csv.py        # tables 2, 4, 6 and 10
python3 classify_failures.py           # tables 5 and 11
python3 table1_recompute_v2.py         # table 1
python3 make_appendixB1_fairness.py    # the three tests of appendix B.1; the footnote of section 4 on the best learning rate
python3 make_inventory_from_reported.py  # table 12
```

reproduces those tables from the records alone. The scripts that need archive 1,
archive 2 or a GPU are marked in `MANIFEST.md`.

The scripts print the tables and draw the figures. Other numbers that the text of the
paper quotes (for example p-values, ranges and counts) follow from the columns of
`data/runs_reported.csv`; not every one of them has a script.

To train a run again, see the section "Re-training" of `MANIFEST.md`: it names, for each
row of table 12, the driver in `code/experiments/` and its arguments.

Figure 6 is drawn from the 21 saved fields in `data/derived/appD_frames/`
(needs scikit-image):

```
python3 make_fig_relaxation.py ../../data/derived/appD_frames fig_relaxation.png
```

The environment is in `MANIFEST.md`. The analysis scripts need Python 3 with
numpy, scipy and matplotlib; the five scripts listed first need nothing beyond the
standard library. Re-training needs PyTorch and a GPU.

## 4. Three things to know before using the records

`data/runs_reported_README.md` explains every column of the CSV.

**The three seeds of each pykan configuration are one run repeated.** pykan 0.2.8
calls `torch.manual_seed` in its constructor, and the wrapper in `models.py` did not
pass the seed when these runs were made, so the seed had no effect. Each of the five
(charge, width, depth) configurations of the fifteen pykan runs repeats one run three
times, with the same initialisation and the same quadrature draws; its three energies
agree to 2.2e-7 or better. No result of the paper uses them as independent seeds:
figure 2 draws no pykan series, and appendix B.2 uses them for timing. When you compute a
spread over seeds from the records, leave the pykan runs out.

**Three source files changed during the project.** `physics.py` gained
fixed-seed restarts of the profile pre-optimisation on 2026-09-07, and the
threshold for preferring a restart rose from 1e-9 to 1e-6 on 2026-09-08.
`train.py` gained the early stopping of the imprinting regression on
2026-09-14; only the runs of table 8 use it. `MANIFEST.md` says which runs each version produced, and
`code/src/history/` holds the earlier versions of these two files. The anchored
runs of seeds 3 and 4 at B = 2 were trained again with the corrected profile
(`2026-09-11_refit_v2/runs_refit_v2`, 96 runs). In `models.py`, the pykan wrapper now passes the seed (see above).

**A re-run does not always reproduce a record bit for bit.** Section 7.4 of the
paper reports the measurement: of thirteen re-runs with the original seeds, nine
reproduce the record bit for bit, and the other four differ in the energy by
2.1e-8 (two runs), 1.6e-5 and 1.5e-4 (a width-8 network). The anchored-track
drivers train the 1D profile once before the first run
(`torch.manual_seed(0); get_base_profile(B, device)`); without this step the first
anchored run of a process differs from its record.

## 5. License

The code and the data in this package are licensed under CC BY 4.0 (see `LICENSE`). When you use them, please cite the paper.

The EffKAN layer follows the design
of efficient-kan (MIT); pykan (MIT) is used only as the reference implementation
of appendix B.2.
