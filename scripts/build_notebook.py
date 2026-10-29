"""
build_notebook.py -- writes notebooks/01_explore_mitdb.ipynb and executes it, saving outputs into
the notebook so it can be read on GitHub without running it.

    python scripts/build_notebook.py
"""
import base64
import contextlib
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

CELLS = [
("md", """# Milestone 1: exploring the MIT-BIH Arrhythmia Database

*Don't Skip a Beat* (STAT GU4243, Fall 2026). Abhishek Shanmuga Vengatasalam and Sneha Sivakumar.

This notebook walks through every step behind the Data section of our Milestone 1 report, one small
piece at a time. It uses the functions in `src/ecg_data.py`; the full set of report figures is made by
`scripts/make_m1_figures.py` (last cell).

**Run it:** open it from the `notebooks/` folder and run the cells from top to bottom. The first cell
downloads the data from PhysioNet if it is not already in `data/mitdb/`."""),
("code", """import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))   # lets us import ecg_data
import ecg_data as E

DATA = os.path.join("..", "data", "mitdb")
if not os.path.exists(DATA):
    E.download_mitdb(DATA)        # ~100 MB; or unzip PhysioNet's ZIP into data/mitdb/
FOLDER = E.find_record_dir(DATA)
print("records found:", len([f for f in os.listdir(FOLDER) if f.endswith(".hea")]))"""),
("md", """## 1. One record: header, signal, annotations

Each record has three files: `.hea` (header: sampling rate, gain, lead names), `.dat` (the signal,
packed as 12-bit integers), and `.atr` (the cardiologists' annotations: one label per beat, placed at
the R peak). `verify_record` checks that our decoder reproduces the checksum stored in the header."""),
("code", """h = E.read_header(os.path.join(FOLDER, "100"))
print(f"record {h.record}: leads {h.leads}, {h.fs:.0f} Hz, {h.n_samples:,} samples = {h.n_samples / h.fs / 60:.1f} min")
print("integrity check:", E.verify_record(FOLDER, 100))

rec = E.load_record(FOLDER, 100)
print("labels in record 100:", pd.Series(rec.ann_symbols).value_counts().to_dict())"""),
("code", """# 6 seconds of lead MLII, before and after the 0.5-40 Hz band-pass filter
fs = rec.fs
seg = slice(0, int(6 * fs))
raw = rec.signal[seg, E.mlii_index(rec)]
filt = E.bandpass(rec.signal[:, E.mlii_index(rec)], fs)[seg]
t = np.arange(len(raw)) / fs

fig, ax = plt.subplots(figsize=(9, 2.6))
ax.plot(t, raw, color="#b5b3ad", lw=1, label="raw")
ax.plot(t, filt, color="#0b0b0b", lw=1, label="band-passed 0.5-40 Hz")
m = rec.ann_samples < len(raw)
for s, y in zip(rec.ann_samples[m], rec.ann_symbols[m]):
    ax.text(s / fs, filt.max() + 0.3, y, ha="center", fontsize=9)
ax.set_xlabel("time (s)"); ax.set_ylabel("mV"); ax.legend(loc="lower right", frameon=False)
ax.set_ylim(filt.min() - 0.5, filt.max() + 0.8)
plt.show()"""),
("md", """## 2. AAMI classes and the inter-patient split

MIT-BIH uses about 20 beat labels. The AAMI EC57 standard groups them into five classes:
**N** (normal and bundle branch block), **S** (supraventricular ectopic), **V** (ventricular ectopic),
**F** (fusion) and **Q** (unknown/paced). The four paced records are excluded. The remaining 44 records
are split by *patient* into DS1 (our unlabelled training pool) and DS2 (test), following
de Chazal et al. (2004), so no person appears in both."""),
("code", """counts = E.aami_counts(FOLDER, E.DS1 + E.DS2)
table = pd.DataFrame(counts).T
table["split"] = ["DS1" if r in E.DS1 else "DS2" for r in table.index]
by_split = table.groupby("split")[E.AAMI_CLASSES].sum()
by_split["total"] = by_split.sum(axis=1)
print(by_split)
print()
print((100 * by_split[E.AAMI_CLASSES].div(by_split["total"], axis=0)).round(2).astype(str) + " %")"""),
("code", """# Rare classes come from a few patients: the records with the most S and F beats
for split, recs in [("DS1", E.DS1), ("DS2", E.DS2)]:
    sub = table.loc[recs]
    for k in ["S", "F"]:
        top = sub[k].sort_values(ascending=False).head(3)
        share = 100 * top / sub[k].sum()
        print(f"{split} {k}: " + ", ".join(f"record {r} has {n} ({p:.0f}%)" for r, n, p in zip(top.index, top, share)))"""),
("md", """## 3. Cutting one window per beat

For every labelled beat we cut 256 samples around the R peak (0.25 s before, 0.46 s after) from lead
MLII and compute RR-interval features. These windows are what the VAE will learn from."""),
("code", """B1 = E.beats_for_records(FOLDER, E.DS1)
print("DS1 beats:", B1["X"].shape, "  classes:", pd.Series(B1["aami"]).value_counts().to_dict())

rng = np.random.default_rng(0)
tb = (np.arange(256) - 90) / 360 * 1000
fig, axs = plt.subplots(1, 4, figsize=(11, 2.4), sharey=True)
for ax, k, c in zip(axs, ["N", "S", "V", "F"], ["#52514e", "#2a78d6", "#eb6834", "#1baf7a"]):
    idx = rng.choice(np.where(B1["aami"] == k)[0], 30, replace=False)
    ax.plot(tb, B1["X"][idx].T, color=c, lw=0.6, alpha=0.5)
    ax.set_title(f"{k}: 30 random beats"); ax.set_xlabel("ms from R peak")
axs[0].set_ylabel("mV")
plt.show()"""),
("md", """## 4. Timing: S beats come early

`rr_ratio` divides the interval before a beat by the mean of the ten intervals before it;
`rr_ratio_long` divides it by the median interval in the surrounding 10 minutes. A ratio below 1
means the beat came early. Neither feature uses labels."""),
("code", """rr = pd.DataFrame({"class": B1["aami"], "rr_ratio": B1["rr_ratio"], "rr_ratio_long": B1["rr_ratio_long"]})
summary = rr.groupby("class").agg(median_local=("rr_ratio", "median"), median_long=("rr_ratio_long", "median"),
                                  early_local=("rr_ratio", lambda v: (v < 0.9).mean()),
                                  early_long=("rr_ratio_long", lambda v: (v < 0.9).mean()))
print(summary.loc[["N", "S", "V", "F"]].round(2))"""),
("md", """## 5. How many dimensions does a beat need?

PCA on beats normalized to zero mean and unit variance. If a few components explain most of the
variance, a low-dimensional latent space (like a VAE's) can describe beat shape well."""),
("code", """from sklearn.decomposition import PCA
Xz = (B1["X"] - B1["X"].mean(1, keepdims=True)) / (B1["X"].std(1, keepdims=True) + 1e-6)
cev = np.cumsum(PCA().fit(Xz).explained_variance_ratio_)
for q in (0.80, 0.90, 0.95, 0.99):
    print(f"{q:.0%} of variance with {np.searchsorted(cev, q) + 1} components")"""),
("md", """## 6. Why the patient split matters

The same simple classifier (1-nearest neighbour on 20 principal components) is scored two ways:
with beats split at random, so training and test share patients, and with the patient-disjoint
DS1 to DS2 split. The difference is how much a random split overstates performance on new patients."""),
("code", """from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B2 = E.beats_for_records(FOLDER, E.DS2)
def feats(B):
    keep = np.isin(B["aami"], ["N", "S", "V", "F"])
    X = np.hstack([B["X"], np.c_[B["rr_pre"], B["rr_post"], B["rr_ratio"]]])
    return X[keep], B["aami"][keep]
def model():
    return make_pipeline(StandardScaler(), PCA(20, random_state=0), KNeighborsClassifier(1))
def report(name, y, p):
    sens = recall_score(y, p, labels=["N", "S", "V", "F"], average=None)
    print(f"{name:28s} macro-F1 {f1_score(y, p, average='macro'):.2f}  sensitivity N/S/V/F " + " ".join(f"{v:.2f}" for v in sens))

X1, y1 = feats(B1); X2, y2 = feats(B2)
Xa, ya = np.vstack([X1, X2]), np.concatenate([y1, y2])
Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.5, stratify=ya, random_state=0)
report("random beat split", yte, model().fit(Xtr, ytr).predict(Xte))
report("patient split DS1 -> DS2", y2, model().fit(X1, y1).predict(X2))"""),
("md", """## 7. Regenerate every figure and number in the report

This runs the same code end to end (about a minute) and writes `figures/` and `results/m1_stats.json`.
`scripts/write_numbers_tex.py` then turns the statistics into the numbers printed in the paper."""),
("code", """import subprocess
for script in ["make_m1_figures.py", "write_numbers_tex.py"]:
    args = [sys.executable, os.path.join("..", "scripts", script)]
    if script == "make_m1_figures.py":
        args += ["--data", DATA]
    print(subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip())"""),
]


def run_and_build():
    os.chdir(NB_DIR)
    ns = {}
    nb_cells = []
    count = 0
    shown = []
    plt.show = lambda *a, **k: shown.append(plt.gcf())
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
            continue
        count += 1
        buf = io.StringIO()
        shown.clear()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"cell{count}", "exec"), ns)
        outputs = []
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": buf.getvalue().splitlines(keepends=True)})
        for fig in shown:
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=90, bbox_inches="tight")
            plt.close(fig)
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                                     "text/plain": ["<Figure>"]}})
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                         "outputs": outputs, "source": lines})
        print(f"cell {count} ok")
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                           "name": "python3"},
                                           "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(NB_DIR, "01_explore_mitdb.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    run_and_build()
