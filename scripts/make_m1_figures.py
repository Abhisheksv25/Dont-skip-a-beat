import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import ecg_data as E  # noqa: E402

COL = {"N": "#a8a6a0", "S": "#2a78d6", "V": "#eb6834", "F": "#1baf7a", "Q": "#52514e"}
MARK = {"N": "o", "S": "s", "V": "^", "F": "D"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.facecolor": "white",
})
TEXTWIDTH = 5.5  


def _num(v, _=None):
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def log_ticks(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10))
    a.set_major_formatter(FuncFormatter(_num))
    a.set_minor_formatter(NullFormatter())


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontsize=8, fontweight="bold", color=INK, pad=4)


def save(fig, name, figdir):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


DOC_BEAT_TOTALS = {100: 2273, 101: 1865, 102: 2187, 103: 2084, 104: 2229, 105: 2572,
                   106: 2027, 107: 2137, 108: 1774, 109: 2532, 111: 2124, 112: 2539,
                   113: 1795, 114: 1879, 115: 1953, 116: 2412, 117: 1535, 118: 2288,
                   119: 1987, 121: 1863, 122: 2476, 123: 1518, 124: 1619, 200: 2601,
                   201: 2000, 202: 2136, 203: 2980, 205: 2656, 207: 2332, 208: 2955,
                   209: 3005, 210: 2650, 212: 2748, 213: 3251, 214: 2262, 215: 3363,
                   217: 2208, 219: 2287, 220: 2048, 221: 2427}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "mitdb"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--resdir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.resdir, exist_ok=True)
    if args.download:
        E.download_mitdb(args.data)
    folder = E.find_record_dir(args.data)
    rng = np.random.default_rng(args.seed)
    S = {}  # everything quoted in the paper
    t0 = time.time()

    # ---------------- 1. integrity checks ----------------------------------------------
    checks = [E.verify_record(folder, r) for r in E.ALL_RECORDS]
    S["integrity_all_records_ok"] = all(c["initial_value_ok"] and c["checksum_ok"] for c in checks)
    mism = {}
    for r, ref in DOC_BEAT_TOTALS.items():
        _, sym, _ = E.read_annotations(os.path.join(folder, str(r)))
        n = int(sum((s in E.BEAT_SYMBOLS) or (s == "x") for s in sym))
        if n != ref:
            mism[r] = (n, ref)
    S["beat_totals_match_documentation"] = (len(mism) == 0)
    S["beat_total_mismatches"] = {str(k): v for k, v in mism.items()}

    hdr = {r: E.read_header(os.path.join(folder, str(r))) for r in E.ALL_RECORDS}
    S["fs"] = sorted({h.fs for h in hdr.values()})
    S["n_samples"] = sorted({h.n_samples for h in hdr.values()})
    S["duration_min"] = S["n_samples"][0] / S["fs"][0] / 60
    S["records_without_MLII"] = [r for r, h in hdr.items() if "MLII" not in h.leads]
    S["records_MLII_not_first"] = [r for r, h in hdr.items() if "MLII" in h.leads and h.leads[0] != "MLII"]
    S["second_leads"] = {}
    for r, h in hdr.items():
        if r in E.PACED:
            continue
        other = [l for l in h.leads if l != "MLII"][0]
        S["second_leads"][other] = S["second_leads"].get(other, 0) + 1

    # all annotation symbols across the database
    sym_all, n_beats_all, n_noise = {}, 0, 0
    for r in E.ALL_RECORDS:
        _, sym, aux = E.read_annotations(os.path.join(folder, str(r)))
        for s in sym:
            sym_all[s] = sym_all.get(s, 0) + 1
        n_beats_all += int(sum(s in E.BEAT_SYMBOLS for s in sym))
    S["n_beat_annotations_all48"] = n_beats_all
    S["symbol_counts_all48"] = dict(sorted(sym_all.items(), key=lambda kv: -kv[1]))

    # ---------------- 2. class counts per record and per split -------------------------
    counts = E.aami_counts(folder, E.DS1 + E.DS2)
    with open(os.path.join(args.resdir, "class_counts_per_record.csv"), "w") as f:
        f.write("record,split," + ",".join(E.AAMI_CLASSES) + ",total\n")
        for r in E.DS1 + E.DS2:
            c = counts[r]
            f.write(f"{r},{'DS1' if r in E.DS1 else 'DS2'}," +
                    ",".join(str(c[k]) for k in E.AAMI_CLASSES) + f",{sum(c.values())}\n")
    split_counts = {}
    for name, recs in [("DS1", E.DS1), ("DS2", E.DS2)]:
        tot = {k: sum(counts[r][k] for r in recs) for k in E.AAMI_CLASSES}
        n = sum(tot.values())
        split_counts[name] = {"counts": tot, "total": n,
                              "pct": {k: 100 * v / n for k, v in tot.items()}}
        # how concentrated is each rare class in a few patients?
        conc = {}
        for k in ["S", "V", "F"]:
            per = sorted(((counts[r][k], r) for r in recs), reverse=True)
            conc[k] = {"top_record": per[0][1], "top_share_pct": 100 * per[0][0] / max(tot[k], 1),
                       "top3_share_pct": 100 * sum(p[0] for p in per[:3]) / max(tot[k], 1),
                       "n_records_with_any": int(sum(p[0] > 0 for p in per)),
                       "n_records_with_10plus": int(sum(p[0] >= 10 for p in per))}
        split_counts[name]["concentration"] = conc
    S["splits"] = split_counts
    S["imbalance_ratio_N_to_F_DS1"] = split_counts["DS1"]["counts"]["N"] / split_counts["DS1"]["counts"]["F"]

    # random-annotation budget: expected examples of each class in k random DS1 beats
    p = {k: split_counts["DS1"]["pct"][k] / 100 for k in E.AAMI_CLASSES}
    S["random_budget_for_10_examples"] = {k: int(np.ceil(10 / p[k])) for k in ["S", "V", "F"]}
    S["expected_in_500_random"] = {k: 500 * p[k] for k in ["S", "V", "F"]}
    n1 = split_counts["DS1"]["total"]
    S["budget_40pct_DS1"] = int(0.4 * n1)

    # ---------------- 3. beats ----------------------------------------------------------
    B1 = E.beats_for_records(folder, E.DS1)
    B2 = E.beats_for_records(folder, E.DS2)
    S["n_extracted_DS1"], S["n_extracted_DS2"] = len(B1["aami"]), len(B2["aami"])
    S["window_samples"], S["window_sec"] = int(B1["X"].shape[1]), B1["X"].shape[1] / 360
    rr = {}
    for k in ["N", "S", "V", "F"]:
        m = B1["aami"] == k
        rr[k] = {"rr_ratio_median": float(np.median(B1["rr_ratio"][m])),
                 "rr_ratio_iqr": [float(np.percentile(B1["rr_ratio"][m], 25)),
                                  float(np.percentile(B1["rr_ratio"][m], 75))],
                 "rr_pre_median_s": float(np.median(B1["rr_pre"][m])),
                 "frac_rr_ratio_below_0.9": float(np.mean(B1["rr_ratio"][m] < 0.9)),
                 "rr_ratio_long_median": float(np.median(B1["rr_ratio_long"][m])),
                 "frac_rr_ratio_long_below_0.9": float(np.mean(B1["rr_ratio_long"][m] < 0.9))}
    S["rr_DS1"] = rr
    mF = B1["aami"] == "F"
    S["F_context"] = {"median_rr_post_s": float(np.median(B1["rr_post"][mF])),
                      "frac_rr_post_below_0.46s": float(np.mean(B1["rr_post"][mF] < 0.46)),
                      "N_median_rr_post_s": float(np.median(B1["rr_post"][B1["aami"] == "N"]))}
    mS = B1["aami"] == "S"
    late = mS & (B1["rr_ratio"] >= 0.9)
    S["S_not_premature_local"] = {"frac_of_S": float(late.sum() / mS.sum()),
                                  "frac_following_S": float(np.mean(B1["prev_aami"][late] == "S")),
                                  "frac_premature_vs_long_baseline": float(np.mean(B1["rr_ratio_long"][late] < 0.9))}
    hr = 60 / np.array([np.median(B1["rr_pre"][B1["record"] == r]) for r in E.DS1 + []] +
                       [np.median(B2["rr_pre"][B2["record"] == r]) for r in E.DS2])
    S["median_heart_rate_bpm_range_over_records"] = [float(hr.min()), float(hr.max())]

    # ---------------- Figure 1: strip + morphology --------------------------------------
    best = None
    for r in E.DS1:
        rec = E.load_record(folder, r)
        isb = np.isin(rec.ann_symbols, list(E.AAMI))
        t, y = rec.ann_samples[isb], np.array([E.AAMI[s] for s in rec.ann_symbols[isb]])
        win = int(8 * rec.fs)
        for start in range(0, len(rec.signal) - win, int(rec.fs)):
            m = (t >= start + 20) & (t < start + win - 20)
            cl = set(y[m])
            score = (len(cl & {"S", "V", "F"}), -int(np.sum(y[m] != "N")))
            if "N" in cl and (best is None or score > best[0]):
                best = (score, r, start)
    _, strip_rec, strip_start = best
    rec = E.load_record(folder, strip_rec)
    x = E.bandpass(rec.signal[:, E.mlii_index(rec)], rec.fs)
    win = int(8 * rec.fs)
    seg = x[strip_start: strip_start + win]
    tt = np.arange(win) / rec.fs
    isb = np.isin(rec.ann_symbols, list(E.AAMI))
    ta, ya = rec.ann_samples[isb], rec.ann_symbols[isb]
    m = (ta >= strip_start) & (ta < strip_start + win)
    S["strip"] = {"record": strip_rec, "start_s": strip_start / rec.fs,
                  "classes": sorted(set(E.AAMI[s] for s in ya[m]))}

    fig = plt.figure(figsize=(TEXTWIDTH, 3.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.78, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tt, seg, color=INK, lw=0.8)
    top = seg.max() + 0.15 * np.ptp(seg)
    for s, sym in zip(ta[m], ya[m]):
        c = E.AAMI[sym]
        ax.plot((s - strip_start) / rec.fs, top, marker=MARK.get(c, "o"), ms=4.2,
                color=COL[c], mec="white", mew=0.6, ls="none")
        ax.text((s - strip_start) / rec.fs, top + 0.09 * np.ptp(seg), sym, ha="center",
                va="bottom", fontsize=6.3, color=INK2)
    ax.set_ylim(seg.min() - 0.05 * np.ptp(seg), top + 0.3 * np.ptp(seg))
    ax.set_xlim(0, win / rec.fs)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MLII (mV)")
    panel_label(ax, f"(a) Record {strip_rec}, 8-s excerpt")
    handles = [plt.Line2D([], [], marker=MARK[k], color=COL[k], ls="none", ms=4.5, mec="white",
                          mew=0.6, label=f"{k}") for k in ["N", "S", "V", "F"] if k in S["strip"]["classes"]]
    ax.legend(handles=handles, ncol=4, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.98), handletextpad=0.1, columnspacing=0.8, borderaxespad=0)

    tb = (np.arange(B1["X"].shape[1]) - 90) / 360 * 1000
    medN = np.median(B1["X"][B1["aami"] == "N"], axis=0)
    for j, k in enumerate(["N", "S", "V", "F"]):
        ax = fig.add_subplot(gs[1, j])
        Xk = B1["X"][B1["aami"] == k]
        q25, q50, q75 = np.percentile(Xk, [25, 50, 75], axis=0)
        ax.fill_between(tb, q25, q75, color=COL[k], alpha=0.25, lw=0)
        if k != "N":
            ax.plot(tb, medN, color=MUTED, lw=0.9)
        ax.plot(tb, q50, color=COL[k] if k != "N" else INK2, lw=1.3)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_ylim(-1.6, 2.4)
        ax.set_xlabel("ms from R peak")
        if j == 0:
            ax.set_ylabel("mV")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{'(b) ' if j == 0 else ''}{k}  (n={len(Xk):,})", loc="left", fontsize=7.5, color=INK)
    save(fig, "fig1_strip_and_morphology", args.figdir)

    # ---------------- Figure 2: imbalance, random budget, RR ratio ---------------------
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.95), gridspec_kw={"wspace": 0.55})
    ax = axs[0]
    cls = ["N", "S", "V", "F", "Q"]
    for i, k in enumerate(cls):
        for name, filled, dx in [("DS1", True, -0.12), ("DS2", False, 0.12)]:
            v = split_counts[name]["pct"][k]
            ax.plot(i + dx, v, marker="o", ms=5, ls="none", color=COL[k],
                    mfc=COL[k] if filled else "white", mew=1.3)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 200)
    log_ticks(ax)
    ax.set_xticks(range(len(cls)), cls)
    ax.set_xlim(-0.6, len(cls) - 0.4)
    ax.set_ylabel("share of beats (%)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=MUTED, ls="none", ms=5, label="DS1 (train pool)"),
                       plt.Line2D([], [], marker="o", color=MUTED, mfc="white", ls="none", ms=5, mew=1.3,
                                  label="DS2 (test)")],
              frameon=False, loc="lower left", handletextpad=0.1, borderaxespad=0.2)
    panel_label(ax, "(a) Class shares")

    ax = axs[1]
    k_grid = np.logspace(1, np.log10(n1), 200)
    for k in ["S", "V", "F"]:
        ax.plot(k_grid, k_grid * p[k], color=COL[k], lw=1.6)
        ax.text(k_grid[-1] * 1.12, k_grid[-1] * p[k], k, va="center", fontsize=7, color=INK2)
    ax.axhline(10, color=MUTED, lw=0.7)
    ax.text(n1 * 1.6, 7.0, "10 examples", fontsize=6.3, color=INK2, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, n1 * 2.2)
    ax.set_ylim(0.05, 5e3)
    log_ticks(ax, "x")
    log_ticks(ax, "y")
    ax.set_xlabel("random labels bought (beats)")
    ax.set_ylabel("expected labelled examples")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    panel_label(ax, "(b) Random annotation")

    ax = axs[2]
    bins = np.linspace(0.3, 1.8, 61)
    for k in ["F", "V", "S", "N"]:          # N last so it stays visible on top of F
        v = B1["rr_ratio_long"][B1["aami"] == k]
        h, e = np.histogram(v, bins=bins)
        h = h / (len(v) * np.diff(e))        # density over all beats of the class
        c = (e[:-1] + e[1:]) / 2
        ax.plot(c, h, color=COL[k] if k != "N" else INK2, lw=1.0 if k == "N" else 1.4)
        ax.fill_between(c, 0, h, color=COL[k], alpha=0.10, lw=0)
    ax.legend(handles=[plt.Line2D([], [], color=COL[k] if k != "N" else INK2, lw=1.6, label=k)
                       for k in ["N", "S", "V", "F"]], frameon=False, loc="upper right",
              handlelength=1.2, borderaxespad=0)
    ax.axvline(1.0, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("pre-RR / 10-min median RR")
    ax.set_ylabel("density")
    ax.set_xlim(0.3, 1.8)
    panel_label(ax, "(c) Prematurity")
    save(fig, "fig2_imbalance_budget_rr", args.figdir)

    # ---------------- Figure 3: per-record class composition --------------------------
    recs = E.DS1 + E.DS2
    M = np.array([[counts[r][k] for r in recs] for k in ["N", "S", "V", "F"]], dtype=float)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.55))
    im = ax.imshow(np.log10(M + 1), aspect="auto", cmap=BLUES, vmin=0, vmax=np.log10(M.max() + 1))
    ax.set_yticks(range(4), ["N", "S", "V", "F"])
    ax.set_xticks(range(len(recs)), [str(r) for r in recs], rotation=90, fontsize=5.6)
    ax.axvline(len(E.DS1) - 0.5, color=INK, lw=1.0)
    ax.text(len(E.DS1) / 2 - 0.5, -0.9, "DS1  (training pool, 22 patients)", ha="center", fontsize=7, color=INK)
    ax.text(len(E.DS1) + len(E.DS2) / 2 - 0.5, -0.9, "DS2  (test, 22 patients)", ha="center", fontsize=7, color=INK)
    for i in range(4):
        for j in range(len(recs)):
            if M[i, j] == 0:
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["1", "10", "100", "1k"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.set_label("beats", fontsize=6.5)
    save(fig, "fig3_per_record_heatmap", args.figdir)

    # ---------------- 4. structure: PCA, t-SNE, patient signal, leakage ----------------
    Xz1 = (B1["X"] - B1["X"].mean(1, keepdims=True)) / (B1["X"].std(1, keepdims=True) + 1e-6)
    pca_full = PCA().fit(Xz1)
    cev = np.cumsum(pca_full.explained_variance_ratio_)
    S["pca_components_for"] = {f"{q}": int(np.searchsorted(cev, q / 100) + 1) for q in (80, 90, 95, 99)}
    S["pca_cev_first10"] = [float(v) for v in cev[:10]]

    # subsample for t-SNE
    idx = []
    for k, cap in [("N", 3000), ("S", 10 ** 9), ("V", 1500), ("F", 10 ** 9)]:
        ii = np.where(B1["aami"] == k)[0]
        idx.append(rng.choice(ii, size=min(cap, len(ii)), replace=False))
    idx = np.concatenate(idx)
    Z30 = PCA(30, random_state=args.seed).fit_transform(Xz1[idx])
    emb = TSNE(2, perplexity=40, init="pca", random_state=args.seed).fit_transform(Z30)
    S["tsne_points"] = {k: int(np.sum(B1["aami"][idx] == k)) for k in ["N", "S", "V", "F"]}
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(Z30)
    _, nbr = nn.kneighbors(Z30)
    nbr = nbr[:, 1:]
    same_rec = (B1["record"][idx][nbr] == B1["record"][idx][:, None]).mean(1)
    same_cls = (B1["aami"][idx][nbr] == B1["aami"][idx][:, None]).mean(1)
    S["neighbour_purity_pca30"] = {"same_patient": float(same_rec.mean()), "same_class": float(same_cls.mean()),
                                   "same_patient_S": float(same_rec[B1["aami"][idx] == "S"].mean()),
                                   "same_class_S": float(same_cls[B1["aami"][idx] == "S"].mean())}
    # what the same-patient / same-class rates would be if neighbours were drawn at random
    _, pc = np.unique(B1["record"][idx], return_counts=True)
    _, cc = np.unique(B1["aami"][idx], return_counts=True)
    S["neighbour_purity_pca30"]["chance_same_patient"] = float(np.sum((pc / pc.sum()) ** 2))
    S["neighbour_purity_pca30"]["chance_same_class"] = float(np.sum((cc / cc.sum()) ** 2))

    # patient identifiability from a single normal beat (within DS1)
    mN = np.where(B1["aami"] == "N")[0]
    pid_idx = rng.choice(mN, size=min(20000, len(mN)), replace=False)
    yp = B1["record"][pid_idx]
    keep = np.isin(yp, [r for r in np.unique(yp) if np.sum(yp == r) >= 5])
    pipe = make_pipeline(StandardScaler(), PCA(20, random_state=args.seed),
                         LogisticRegression(max_iter=2000))
    acc = cross_val_score(pipe, B1["X"][pid_idx][keep], yp[keep],
                          cv=StratifiedKFold(5, shuffle=True, random_state=args.seed))
    vals, cts = np.unique(yp[keep], return_counts=True)
    S["patient_id_from_normal_beat"] = {"accuracy_mean": float(acc.mean()), "accuracy_sd": float(acc.std()),
                                        "n_patients": int(len(vals)),
                                        "chance_majority": float(cts.max() / cts.sum()),
                                        "chance_uniform": float(1 / len(vals)), "n_beats": int(keep.sum())}

    # leakage check: the same simple classifier under a beat-level random split vs the
    # patient-disjoint DS1 -> DS2 split
    def feats(B):
        return np.hstack([B["X"], np.c_[B["rr_pre"], B["rr_post"], B["rr_ratio"]] * [1, 1, 1]])

    keep4 = lambda B: np.isin(B["aami"], ["N", "S", "V", "F"])
    k1, k2 = keep4(B1), keep4(B2)
    Xa, ya = np.vstack([feats(B1)[k1], feats(B2)[k2]]), np.concatenate([B1["aami"][k1], B2["aami"][k2]])

    def model():
        return make_pipeline(StandardScaler(), PCA(20, random_state=args.seed),
                             KNeighborsClassifier(n_neighbors=1))

    def scores(ytrue, ypred):
        return {"macro_f1": float(f1_score(ytrue, ypred, average="macro", labels=["N", "S", "V", "F"])),
                **{f"sens_{k}": float(recall_score(ytrue, ypred, labels=[k], average="macro"))
                   for k in ["N", "S", "V", "F"]}}

    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.5, stratify=ya, random_state=args.seed)
    leak = {"random_beat_split": scores(yte, model().fit(Xtr, ytr).predict(Xte))}
    pred2 = model().fit(feats(B1)[k1], B1["aami"][k1]).predict(feats(B2)[k2])
    y2, r2 = B2["aami"][k2], B2["record"][k2]
    leak["patient_split_DS1_to_DS2"] = scores(y2, pred2)
    missedS = (y2 == "S") & (pred2 != "S")
    top = S["splits"]["DS2"]["concentration"]["S"]["top_record"]
    leak["patient_split_S_breakdown"] = {
        "missed_S_called_N": float(np.mean(pred2[missedS] == "N")),
        "top_record": top,
        "sens_S_top_record": float(np.mean(pred2[(y2 == "S") & (r2 == top)] == "S")),
        "sens_S_other_records": float(np.mean(pred2[(y2 == "S") & (r2 != top)] == "S"))}
    S["leakage_check_1nn"] = leak

    # ---------------- Figure 4 --------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.05),
                            gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1.1, 1.15]})
    ax = axs[0]
    nshow = 40
    ax.plot(np.arange(1, nshow + 1), 100 * cev[:nshow], color=INK2, lw=1.6)
    for q, dx, dy in ((90, 6, -32), (95, 9, -16)):
        c = S["pca_components_for"][str(q)]
        ax.plot(c, 100 * cev[c - 1], marker="o", ms=4.5, color=INK, mec="white", mew=0.8)
        ax.annotate(f"{q}% with {c} PCs", (c, 100 * cev[c - 1]), xytext=(c + dx, 100 * cev[c - 1] + dy),
                    fontsize=6.3, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("principal components")
    ax.set_ylabel("variance explained (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, nshow + 1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    panel_label(ax, "(a) Low-dimensional")

    ax = axs[1]
    yy = B1["aami"][idx]
    for k in ["N", "V", "F", "S"]:
        mm = yy == k
        ax.scatter(emb[mm, 0], emb[mm, 1], s=2.2 if k == "N" else 3.5, marker=MARK[k], c=COL[k],
                   linewidths=0, alpha=0.6 if k == "N" else 0.85, rasterized=True, label=k)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], marker=MARK[k], color=COL[k], ls="none", ms=4.2, label=k)
                       for k in ["N", "S", "V", "F"]], frameon=False, loc="lower left",
              bbox_to_anchor=(-0.02, -0.2), ncol=4, handletextpad=0.05, columnspacing=0.6)
    panel_label(ax, "(b) t-SNE of beat shape")

    ax = axs[2]
    labels = ["macro-F1", "sens. S", "sens. V", "sens. F"]
    keys = ["macro_f1", "sens_S", "sens_V", "sens_F"]
    r_, p_ = leak["random_beat_split"], leak["patient_split_DS1_to_DS2"]
    for i, kk in enumerate(keys):
        y = 3 - i
        ax.plot([p_[kk], r_[kk]], [y, y], color=GRID, lw=2.0, solid_capstyle="round", zorder=1)
        ax.plot(r_[kk], y, marker="o", ms=5.5, color="#c9c7c1", mec="white", mew=0.8, zorder=2)
        ax.plot(p_[kk], y, marker="o", ms=5.5, color="#2b2b29", mec="white", mew=0.8, zorder=3)
        ax.text(p_[kk], y + 0.22, f"{p_[kk]:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK)
        ax.text(r_[kk], y - 0.22, f"{r_[kk]:.2f}", ha="center", va="top", fontsize=6.2, color=INK2)
    ax.set_yticks(range(4), labels[::-1])
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(-0.08, 1.08)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("score on held-out beats")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="#c9c7c1", ls="none", ms=5.5, label="random beat split"),
                       plt.Line2D([], [], marker="o", color="#2b2b29", ls="none", ms=5.5,
                                  label="patient split (DS1→DS2)")],
              frameon=False, loc="upper left", bbox_to_anchor=(-0.05, -0.27), ncol=1, handletextpad=0.1)
    panel_label(ax, "(c) Same 1-NN, two splits")
    save(fig, "fig4_structure_and_leakage", args.figdir)

    S["runtime_s"] = time.time() - t0
    with open(os.path.join(args.resdir, "m1_stats.json"), "w") as f:
        json.dump(S, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: S[k] for k in ["integrity_all_records_ok", "beat_totals_match_documentation"]}, indent=1))
    print("done in", round(S["runtime_s"]), "s")


if __name__ == "__main__":
    main()
