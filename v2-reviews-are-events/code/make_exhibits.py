"""Turn simulation output and the pilot data into the manuscript's figures (vector PDF), tables and macros.

Run from code/:   python3 make_exhibits.py [--pilot path/to/place-month-Vermont.csv]
Nothing in the manuscript is typed by hand: every number is written here to ../tables/*.tex.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

from vrindex import (RatingScaleEB, family_probs, _invert_mean, drift_interval, breakdown, contamination_interval, simulate_class)

OUT, FIG, TAB = Path("../output"), Path("../figures"), Path("../tables")
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
NU = np.arange(1, 6.0)

# ---- figure style: sans-serif (Arial-metric), thin marks, recessive axes, validated three-hue palette ----------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8, "pdf.fonttype": 42, "lines.linewidth": 1.6, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
MACROS = {}


def macro(name, value):
    MACROS[name] = value


def panel(ax, tag):
    ax.set_title(tag, loc="left", fontweight="bold", pad=4)


def load(name):
    p = OUT / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


# ------------------------------------------------------------------------------------------------------
# Figure 1 and Table 1: the identification argument in one example (population quantities, no sampling)
# ------------------------------------------------------------------------------------------------------
def figure_identification():
    c = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35]))
    b = np.array([15, 2, 1, 1.5, 5.0])
    theta = np.array([-0.5, 0.0, 0.5])
    pi = family_probs(theta, c, NU)
    vis = pi * b; vis /= vis.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), gridspec_kw=dict(wspace=0.5))
    cols, marks, names = [ORANGE, MUTED, BLUE], ["v", "s", "o"], ["Unit A", "Unit B", "Unit C"]
    for ax, P, ttl, yl in ((axes[0], pi, "(a) All encounters", "Share of encounters"),
                           (axes[1], vis, "(b) Visible ratings", "Share of ratings")):
        for j in range(3):
            ax.plot(NU, P[j], color=cols[j], marker=marks[j], mec="white", mew=0.8, label=names[j])
        ax.set_ylim(0, 0.9); ax.set_xticks(NU); ax.set_xlabel("Stars"); ax.set_ylabel(yl); panel(ax, ttl)
    axes[0].legend(loc="upper left", handlelength=1.6, borderaxespad=0.1)
    ax = axes[2]
    for j in (0, 2):
        lat = np.log(pi[j] / pi[1]); v = np.log(vis[j] / vis[1])
        ax.plot(NU, lat - lat[2], color=cols[j], lw=1.2)
        ax.plot(NU, v - v[2], ls="none", marker=marks[j], color=cols[j], mec="white", mew=0.8)
    ax.text(4.75, 0.55, "C vs. B", ha="left", va="center", fontsize=8); ax.text(4.75, -0.55, "A vs. B", ha="left", va="center", fontsize=8)
    ax.set_xlim(0.7, 5.9); ax.set_ylim(-1.25, 1.25); ax.set_xticks(NU); ax.set_xlabel("Stars"); ax.set_ylabel("Log ratio to unit B (centred)")
    panel(ax, "(c) Log ratios")
    fig.savefig(FIG / "fig1-identification.pdf"); plt.close(fig)
    macro("ExLatentMeans", ", ".join(f"{m:.2f}" for m in pi @ NU))
    macro("ExVisibleMeans", ", ".join(f"{m:.2f}" for m in vis @ NU))
    macro("ExLatentB", f"{(pi @ NU)[1]:.2f}"); macro("ExLatentC", f"{(pi @ NU)[2]:.2f}")

    # Table 1: the same three units under three voice functions
    from scipy.optimize import minimize
    from scipy.special import expit
    def cumlogit(P):
        def nll(par):
            kap = np.cumsum(np.concatenate([[par[0]], np.exp(par[1:4])])); loc = np.concatenate([[0.0], par[4:]])
            cdf = expit(kap[None, :] - loc[:, None]); cdf = np.concatenate([np.zeros((3, 1)), cdf, np.ones((3, 1))], 1)
            return -(P * np.log(np.clip(np.diff(cdf, axis=1), 1e-300, None))).sum()
        r = minimize(nll, np.array([-2, 0, 0, 0, 0, 0.0]), method="BFGS"); return np.concatenate([[0.0], r.x[4:]])
    voices = [("None (census)", np.ones(5)), ("Negativity", np.array([6, 3, 1.5, 1, 1.0])), ("J-shaped", b)]
    rows = []
    for name, bb in voices:
        v = pi * bb; v /= v.sum(1, keepdims=True)
        psi = np.log(v[1]); psi -= psi[0]
        th = np.array([_invert_mean(x @ NU, psi, NU) for x in v])
        cl = cumlogit(v)
        rows.append((name, np.diff(v @ NU), np.diff(v[:, 3:].sum(1)), np.diff(cl), np.diff(th)))
    lines = [r"\begin{tabular}{@{}l cc cc cc cc@{}}\toprule",
             r" & \multicolumn{2}{c}{Mean stars} & \multicolumn{2}{c}{Share 4--5 stars} & \multicolumn{2}{c}{Cumulative logit} & \multicolumn{2}{c}{Index $\theta$}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
             r"Voice function & B$-$A & C$-$B & B$-$A & C$-$B & B$-$A & C$-$B & B$-$A & C$-$B\\\midrule"]
    for name, a1, a2, a3, a4 in rows:
        lines.append(f"{name} & " + " & ".join(f"{x:.2f}" for pair in (a1, a2, a3, a4) for x in pair) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (TAB / "tab1-example.tex").write_text("\n".join(lines) + "\n")
    macro("ExStarsNone", f"{rows[0][1][0]:.2f}"); macro("ExStarsJ", f"{rows[2][1][0]:.2f}")


# ------------------------------------------------------------------------------------------------------
# Population-level numbers quoted in the text (sensitivity analysis, counterexamples); no sampling involved
# ------------------------------------------------------------------------------------------------------
def theory_numbers():
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp
    c = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35]))
    bJ = np.array([15, 2, 1, 1.5, 5.0]); bN = np.array([6, 3, 1.5, 1, 1.0])
    psi = c + np.log(bJ); psi -= psi[0]

    def kappa(th):
        p = family_probs(np.array(th), psi, NU); m = p @ NU
        return max(((NU - m) * p)[k:].sum() for k in range(1, 5)) / (p @ (NU - m) ** 2)
    lo, hi = drift_interval(0.0, psi, NU, 2.0)
    macro("DriftHalfExact", f"{max(-lo, hi):.3f}"); macro("DriftHalfRule", f"{np.log(2) / 4:.3f}")
    lo, hi = drift_interval(0.5, psi, NU, 2.0)
    macro("DriftHalfHigh", f"{max(0.5 - lo, hi - 0.5):.2f}")
    macro("KappaCentre", f"{kappa(0.0):.2f}"); macro("KappaHigh", f"{kappa(0.5):.2f}")
    macro("BreakQuarter", f"{breakdown(0.125, -0.125, psi, NU):.1f}"); macro("BreakHalf", f"{breakdown(0.25, -0.25, psi, NU):.1f}")
    macro("BreakHalfHigh", f"{breakdown(0.75, 0.25, psi, NU):.1f}")
    macro("BreakQuarterBound", f"{np.exp(0.5):.2f}"); macro("BreakHalfBound", f"{np.exp(1.0):.2f}")
    # non-genuine ratings: mean-based bounds versus the set that uses the whole histogram
    mlo, mhi = contamination_interval(0.0, psi, NU, 0.2)
    grid = np.linspace(-3, 3, 60001); P = family_probs(grid, psi, NU); q = family_probs(np.array(0.0), psi, NU)
    ok = grid[((1 - 0.2) * P <= q[None, :] + 1e-15).all(1)]
    macro("ContamMean", f"[{mlo:.2f}, {mhi:.2f}]".replace("-", "$-$")); macro("ContamHist", f"[{ok.min():.2f}, {ok.max():.2f}]".replace("-", "$-$"))
    # a latent law that is exactly cumulative-logit: fitted location differences under the three voice functions
    kap0 = np.array([-2.9, -1.9, -0.9, 0.6]); loc0 = np.array([0.0, 0.5, 1.0])
    cdf0 = expit(kap0[None, :] - loc0[:, None]); pi_cl = np.diff(np.concatenate([np.zeros((3, 1)), cdf0, np.ones((3, 1))], 1), axis=1)

    def cumlogit(Pm):
        def nll(par):
            kap = np.cumsum(np.concatenate([[par[0]], np.exp(par[1:4])])); loc = np.concatenate([[0.0], par[4:]])
            cdf = expit(kap[None, :] - loc[:, None]); cdf = np.concatenate([np.zeros((3, 1)), cdf, np.ones((3, 1))], 1)
            return -(Pm * np.log(np.clip(np.diff(cdf, axis=1), 1e-300, None))).sum()
        r = minimize(nll, np.array([-2, 0, 0, 0, 0, 0.0]), method="BFGS"); return np.diff(np.concatenate([[0.0], r.x[4:]]))
    for tag, bb in (("Neg", bN), ("J", bJ)):
        v = pi_cl * bb; v /= v.sum(1, keepdims=True); d = cumlogit(v)
        macro(f"CumExact{tag}", f"{d[0]:.2f} and {d[1]:.2f}")
    # polarisation example (Supplement E)
    quad = (NU - NU.mean()) ** 2
    for tag, (th, ph) in (("Moderate", (0.15, -0.35)), ("Polar", (0.0, 0.35))):
        e = c + th * NU + ph * quad; pl_ = np.exp(e - logsumexp(e)); v = pl_ * bJ; v /= v.sum()
        macro(f"PolLat{tag}", f"{pl_ @ NU:.2f}"); macro(f"PolVis{tag}", f"{v @ NU:.2f}")
    # index from a rating count and a rounded average (Supplement B)
    d = simulate_class(1500, np.random.default_rng(77), b=bJ)
    X = d["X"]; X = X[X.sum(1) > 0].astype(float)
    fit = RatingScaleEB().fit(X)
    N = X.sum(1); avg = np.round((X @ NU) / N, 1)
    post = fit._posterior(np.atleast_2d(fit.psi_), np.zeros(len(N)), fit.tau_, X[:, None, :], N[:, None], (N * avg)[:, None])
    macro("RoundCorr", f"{np.corrcoef(post['mean'], fit.theta_)[0, 1]:.3f}")
    macro("RoundMaxDev", f"{np.max(np.abs(post['mean'] - fit.theta_)):.2f}")
    inner = (avg > 1.05) & (avg < 4.95)
    macro("RoundMaxDevInner", f"{np.max(np.abs(post['mean'] - fit.theta_)[inner]):.2f}")


# ------------------------------------------------------------------------------------------------------
# Figure 2: invariance of an estimated covariate effect (E1)
# ------------------------------------------------------------------------------------------------------
def figure_invariance():
    r = load("E1")
    if r is None:
        return
    regimes = ["none", "negativity", "J-shaped"]
    est = {"raw": "Mean stars", "bayes": "Bayesian average", "cumulative": "Cumulative logit", "index": "Index (this article)"}
    A = {k: np.array([[x[g][k] for g in regimes] for x in r["results"]]) for k in est}
    conv = np.mean([all(x[g]["converged"] for g in regimes) for x in r["results"]])
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    style = {"raw": (MUTED, "s", "-"), "bayes": (MUTED, "^", "--"), "cumulative": (ORANGE, "D", "-"), "index": (BLUE, "o", "-")}
    xs = np.arange(3)
    for k, lab in est.items():
        ratio = A[k] / A[k][:, [0]].mean()
        m = ratio.mean(0); lo, hi = np.percentile(ratio, [2.5, 97.5], axis=0)
        col, mk, ls = style[k]
        off = {"raw": -0.09, "bayes": -0.03, "cumulative": 0.03, "index": 0.09}[k]
        ax.errorbar(xs + off, m, yerr=[m - lo, hi - m], color=col, marker=mk, ls=ls, mec="white", mew=0.8, capsize=0,
                    elinewidth=1.0, label=lab, zorder=3 if k == "index" else 2)
        macro(f"EoneRatio{k.capitalize()}J", f"{m[2]:.2f}"); macro(f"EoneRatio{k.capitalize()}Neg", f"{m[1]:.2f}")
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(["None\n(census)", "Negativity", "J-shaped"]); ax.set_xlim(-0.4, 2.4)
    ax.set_xlabel("Voice function"); ax.set_ylabel("Estimated effect relative to\nits value without selection")
    ax.legend(loc="upper left", handlelength=2.2)
    fig.savefig(FIG / "fig2-invariance.pdf"); plt.close(fig)
    macro("EoneReps", f"{len(r['results'])}"); macro("EoneJ", f"{r['J']:,}".replace(",", "{,}"))
    for g, tag in zip(regimes, ("None", "Neg", "J")):
        macro(f"EoneIndex{tag}", f"{A['index'][:, regimes.index(g)].mean():.3f}")
        macro(f"EoneIndexSD{tag}", f"{A['index'][:, regimes.index(g)].std(ddof=1):.3f}")
        macro(f"EoneRaw{tag}", f"{A['raw'][:, regimes.index(g)].mean():.2f}")
        macro(f"EoneCum{tag}", f"{A['cumulative'][:, regimes.index(g)].mean():.2f}")
        macro(f"EoneBayes{tag}", f"{A['bayes'][:, regimes.index(g)].mean():.2f}")
    latent = np.mean([x['none']['latent_stars'] for x in r['results']])
    macro("EoneLatentStars", f"{latent:.2f}")
    macro("EoneIndexDev", f"{100 * max(abs(A['index'][:, g].mean() / 0.30 - 1) for g in range(3)):.0f}")
    macro("EoneOverstate", f"{100 * (max(A['raw'][:, 1].mean(), A['raw'][:, 2].mean()) / latent - 1):.0f}")
    macro("EoneConverged", f"{100 * conv:.0f}")


# ------------------------------------------------------------------------------------------------------
# Figure 3: sparse and uneven data (E2)
# ------------------------------------------------------------------------------------------------------
def figure_sparse():
    r = load("E2")
    if r is None:
        return
    res = r["results"]
    keys = ["1-2", "3-5", "6-20", "21-100", "101-inf"]
    labels = ["1–2", "3–5", "6–20", "21–100", ">100"]
    est = {"raw": ("Mean stars", MUTED, "s", "-"), "wilson": ("Wilson lower bound", MUTED, "v", ":"),
           "bayes": ("Bayesian average", ORANGE, "^", "--"), "index": ("Index (this article)", BLUE, "o", "-")}
    share = [np.mean([x["bins"][k]["share"] for x in res if k in x["bins"]]) for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw=dict(wspace=0.35))
    ax = axes[0]
    rmse = [np.mean([x["bins"][b]["rmse_index"] for x in res if b in x["bins"]]) for b in keys]
    tau_true = 0.45
    ax.axhline(tau_true, color=INK, lw=0.6)
    ax.text(4.05, tau_true + 0.012, "no ratings", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.plot(range(5), rmse, color=BLUE, marker="o", mec="white", mew=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n({100 * s:.0f}%)" for l, s in zip(labels, share)])
    ax.set_xlabel("Ratings per unit (share of units)"); ax.set_ylabel(r"Root mean squared error of $\hat\theta_j$")
    ax.set_ylim(0, 0.5); panel(ax, "(a) Accuracy")
    ax = axes[1]
    cov = [np.mean([x["bins"][b]["coverage"] for x in res if b in x["bins"]]) for b in keys]
    ax.axhline(0.95, color=INK, lw=0.6)
    ax.plot(range(5), cov, color=BLUE, marker="o", mec="white", mew=0.8)
    ax.set_ylim(0.85, 1.0); ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n" for l in labels])
    ax.set_xlabel("Ratings per unit"); ax.set_ylabel("Coverage of 95% intervals"); panel(ax, "(b) Calibration")
    for b_, v_ in zip(("A", "B", "C", "D", "E"), rmse):
        macro(f"EtwoRmse{b_}", f"{v_:.2f}")
    fig.savefig(FIG / "fig3-sparse.pdf"); plt.close(fig)
    for k in est:
        macro(f"EtwoRho{k.capitalize()}", f"{np.mean([x['overall'][k] for x in res]):.2f}")
        macro(f"EtwoBottom{k.capitalize()}", f"{np.mean([x['bottom_decile_precision'][k] for x in res]):.2f}")
        macro(f"EtwoRhoSmall{k.capitalize()}", f"{np.mean([x['bins']['1-2'][f'rho_{k}'] for x in res]):.2f}")
    macro("EtwoCoverage", f"{np.mean([x['coverage'] for x in res]):.3f}")
    macro("EtwoCoverageSmall", f"{cov[0]:.3f}"); macro("EtwoCoverageLarge", f"{cov[-1]:.3f}")
    macro("EtwoShareLtFive", f"{100 * np.mean([x['share_lt5'] for x in res]):.0f}")
    macro("EtwoMedianN", f"{np.median([x['median_n'] for x in res]):.0f}")
    macro("EtwoReps", f"{len(res)}")
    macro("EtwoConverged", f"{100 * np.mean([x['converged'] for x in res]):.0f}")


# ------------------------------------------------------------------------------------------------------
# Table 2: several sources (E3) and unit-specific drift (E4)
# ------------------------------------------------------------------------------------------------------
def table_sources_drift():
    r3, r4 = load("E3"), load("E4")
    mcse = 0.0
    if r3 is not None:
        tasks = np.array(r3["tasks"]); res = r3["results"]
        names = [("raw", "Pooled mean stars"), ("bayes", "Pooled Bayesian average"), ("demeaned", "Source-demeaned average"),
                 ("index_pooled", "Index, sources pooled"), ("index_stratified", "Index, stratified by source")]
        lines = [r"\begin{tabular}{@{}lcc@{}}\toprule", r" & \multicolumn{2}{c}{Source mix}\\\cmidrule(l){2-3}",
                 r"Summary & unrelated to $\theta_j$ & related to $\theta_j$\\\midrule"]
        for k, lab in names:
            vals = [np.mean([x[k] for x, t in zip(res, tasks) if abs(t[2] - c) < 1e-9]) for c in (0.0, 0.5)]
            lines.append(f"{lab} & {vals[0]:.2f} & {vals[1]:.2f}" + r"\\")
            macro(f"Ethree{k.replace('_', '').capitalize()}Zero", f"{vals[0]:.2f}"); macro(f"Ethree{k.replace('_', '').capitalize()}Half", f"{vals[1]:.2f}")
        lines.append(r"\bottomrule\end{tabular}")
        (TAB / "tab2a-sources.tex").write_text("\n".join(lines) + "\n")
        macro("EthreeReps", f"{int(len(res) / 2)}")
        mcse = max(np.std([x[k] for x, t in zip(res, tasks) if abs(t[2] - c) < 1e-9], ddof=1) / np.sqrt(len(res) / 2)
                   for k, _ in names for c in (0.0, 0.5))
    if r4 is not None:
        tasks = np.array(r4["tasks"]); res = r4["results"]
        lines = [r"\begin{tabular}{@{}l cccc c cccc@{}}\toprule",
                 r" & \multicolumn{4}{c}{Coverage of $\theta_j$} & & \multicolumn{4}{c}{Share of pairs ordered}\\\cmidrule(lr){2-5}\cmidrule(l){7-10}",
                 r"True drift $\Lambda_0$ & $\Lambda{=}1$ & $1.5$ & $2$ & $3$ & & $\Lambda{=}1$ & $1.5$ & $2$ & $3$\\\midrule"]
        worst = 0.0
        for L0 in (1.0, 1.5, 2.0):
            sel = [x for x, t in zip(res, tasks) if abs(t[2] - L0) < 1e-9]
            cov = [np.mean([x[str(L)]["coverage"] for x in sel]) for L in (1.0, 1.5, 2.0, 3.0)]
            dec = [np.mean([x[str(L)]["declared"] for x in sel]) for L in (1.0, 1.5, 2.0, 3.0)]
            worst = max(worst, max(np.mean([x[str(L)]["wrong_among_declared"] for x in sel]) for L in (1.0, 1.5, 2.0, 3.0) if L >= L0))
            lines.append(f"{L0:g} & " + " & ".join(f"{v:.2f}" for v in cov) + " & & " + " & ".join(f"{v:.2f}" for v in dec) + r"\\")
            for L, v in zip(("One", "OneFive", "Two", "Three"), cov):
                macro(f"EfourCov{str(L0).replace('.', 'p').replace('1p0', 'A').replace('1p5', 'B').replace('2p0', 'C')}{L}", f"{v:.2f}")
        lines.append(r"\bottomrule\end{tabular}")
        (TAB / "tab2b-drift.tex").write_text("\n".join(lines) + "\n")
        for L0 in (1.0, 1.5, 2.0):
            sel = [x for x, t in zip(res, tasks) if abs(t[2] - L0) < 1e-9]
            for L in ("1.0", "1.5", "2.0", "3.0"):
                for kk in ("coverage", "declared"):
                    mcse = max(mcse, np.std([x[L][kk] for x in sel], ddof=1) / np.sqrt(len(sel)))
        macro("TabTwoMCSE", f"{np.ceil(mcse * 1000) / 1000:.3f}")
        macro("EfourWrongMax", f"{100 * worst:.1f}"); macro("EfourReps", f"{int(len(res) / 3)}")      # percent
        macro("EfourWrongText", "none is" if worst == 0 else ("fewer than 0.1 percent are" if worst < 0.001 else f"{100 * worst:.1f} percent are"))
        sel = [x for x, t in zip(res, tasks) if abs(t[2] - 2.0) < 1e-9]
        macro("EfourWrongPlain", f"{100 * np.mean([x['1.0']['wrong_among_declared'] for x in sel]):.1f}")  # percent, Lambda_0 = 2
        macro("EfourWidthOne", f"{np.mean([x['1.0']['width'] for x in sel]):.2f}"); macro("EfourWidthTwo", f"{np.mean([x['2.0']['width'] for x in sel]):.2f}")


# ------------------------------------------------------------------------------------------------------
# Supplement tables (E5-E8)
# ------------------------------------------------------------------------------------------------------
def tables_supplement():
    r5, r6, r7, r8 = load("E5"), load("E6"), load("E7"), load("E8")
    if r5 is not None:
        tasks = np.array(r5["tasks"]); res = r5["results"]
        lines = [r"\begin{tabular}{@{}ll ccc ccc@{}}\toprule",
                 r" & & \multicolumn{3}{c}{Rank correlation with latent mean} & \multicolumn{3}{c}{Population-level stability}\\\cmidrule(lr){3-5}\cmidrule(l){6-8}",
                 r"Latent law & Voice & Mean stars & Bayes avg. & Index & Slope, mean & Slope, index & Rank agreement\\\midrule"]
        for h, lab in ((0.0, "Probit, common scale"), (1.0, "Probit, unit-specific scale")):
            sel = [x for x, t in zip(res, tasks) if abs(t[2] - h) < 1e-9]
            for i, g in enumerate(("none", "negativity", "J-shaped")):
                row = [np.mean([x[g][k] for x in sel]) for k in ("raw", "bayes", "index")]
                extra = ([np.mean([x["pop_slope_mean"] for x in sel]), np.mean([x["pop_slope_index"] for x in sel]),
                          np.mean([x["pop_rank_agree_index"] for x in sel])] if g == "J-shaped" else None)
                lines.append((lab if i == 0 else "") + f" & {g} & " + " & ".join(f"{v:.2f}" for v in row) + " & " +
                             (" & ".join(f"{v:.2f}" for v in extra) if extra else " & & ") + r"\\")
            tag = "Hom" if h == 0 else "Het"
            macro(f"EfiveSlopeMean{tag}", f"{np.mean([x['pop_slope_mean'] for x in sel]):.2f}")
            macro(f"EfiveSlopeIndex{tag}", f"{np.mean([x['pop_slope_index'] for x in sel]):.2f}")
            macro(f"EfiveRankAgree{tag}", f"{np.mean([x['pop_rank_agree_index'] for x in sel]):.2f}")
        lines.append(r"\bottomrule\end{tabular}")
        (TAB / "tabS1-misspecification.tex").write_text("\n".join(lines) + "\n")
        m5 = 0.0
        for h in (0.0, 1.0):
            sel = [x for x, t in zip(res, tasks) if abs(t[2] - h) < 1e-9]
            for g in ("none", "negativity", "J-shaped"):
                for k in ("raw", "bayes", "index"):
                    m5 = max(m5, np.std([x[g][k] for x in sel], ddof=1) / np.sqrt(len(sel)))
            for k in ("pop_slope_mean", "pop_slope_index", "pop_rank_agree_index"):
                m5 = max(m5, np.std([x[k] for x in sel], ddof=1) / np.sqrt(len(sel)))
        macro("EfiveMCSE", f"{np.ceil(m5 * 1000) / 1000:.3f}"); macro("EfiveReps", f"{int(len(res) / 2)}")
    if r6 is not None:
        res = r6["results"]
        lines = [r"\begin{tabular}{@{}lccccccc@{}}\toprule",
                 r"Rating volume & $\hat\gamma$ & SD & Mean SE & Coverage & Realized contrast & $\hat\tau$ & Corr$(\theta_j,\log N_j)$\\\midrule"]
        for k, lab in (("exogenous", r"Unrelated to $\theta_j$"), ("endogenous", "As implied by J-shaped voice")):
            v = {m: np.mean([x[k][m] for x in res]) for m in ("gamma", "gamma_se", "cover_gamma", "truth", "tau", "corr_theta_logN", "rho")}
            sd = np.std([x[k]["gamma"] for x in res], ddof=1)
            lines.append(f"{lab} & {v['gamma']:.3f} & {sd:.3f} & {v['gamma_se']:.3f} & {v['cover_gamma']:.2f} & {v['truth']:.3f} & {v['tau']:.3f} & {v['corr_theta_logN']:+.2f}" + r"\\")
            macro(f"Esix{k.capitalize()}Gamma", f"{v['gamma']:.3f}"); macro(f"Esix{k.capitalize()}Tau", f"{v['tau']:.3f}")
            macro(f"Esix{k.capitalize()}SE", f"{v['gamma_se']:.3f}"); macro(f"Esix{k.capitalize()}SD", f"{sd:.3f}")
            macro(f"Esix{k.capitalize()}Cover", f"{100 * v['cover_gamma']:.0f}")
            macro(f"Esix{k.capitalize()}CoverTau", f"{100 * np.mean([x[k]['cover_tau'] for x in res]):.0f}")
        gv = np.mean([x["endogenous"]["gamma_volume_terms"] for x in res])
        lines.append(r"\quad with a quadratic in $\log N_j$ in the prior mean & " + f"{gv:.3f}" + r" & & & & & & \\")
        if "gamma_interacted" in res[0]["endogenous"]:
            gi = np.mean([x["endogenous"]["gamma_interacted"] for x in res])
            lines.append(r"\quad and its interaction with the covariate (average contrast) & " + f"{gi:.3f}" + r" & & & & & & \\")
            macro("EsixInteracted", f"{gi:.3f}")
        macro("EsixEndogenousExcess", f"{100 * (np.mean([x['endogenous']['gamma'] for x in res]) / 0.30 - 1):.0f}")
        macro("EsixEndogenousExcessTau", f"{100 * (np.mean([x['endogenous']['tau'] for x in res]) / 0.45 - 1):.0f}")
        macro("EsixBiasInSE", f"{(np.mean([x['endogenous']['gamma'] for x in res]) - 0.30) / np.mean([x['endogenous']['gamma_se'] for x in res]):.1f}")
        lines.append(r"\bottomrule\end{tabular}")
        (TAB / "tabS2-volume.tex").write_text("\n".join(lines) + "\n")
        m6 = max(np.std([x[k][m] for x in res], ddof=1) / np.sqrt(len(res)) for k in ("exogenous", "endogenous") for m in ("gamma", "tau"))
        macro("EsixMCSE", f"{np.ceil(m6 * 1000) / 1000:.3f}")
        macro("EsixCorr", f"{np.mean([x['endogenous']['corr_theta_logN'] for x in res]):.2f}")
        macro("EsixReps", f"{len(res)}")
    if r7 is not None:
        res = r7["results"]
        macro("EsevenBias", f"{np.mean([x['bias_fake'] for x in res]):.2f}")
        macro("EsevenCovPlain", f"{np.mean([x['cov_plain'] for x in res]):.2f}")
        macro("EsevenCovRobust", f"{np.mean([x['cov_robust'] for x in res]):.2f}")
    if r8 is not None:
        res = r8["results"]
        for k, tag in (("rho_group", "Group"), ("rho_pooled", "Pooled"), ("rho_unitavg", "UnitAvg"), ("rho_bayesavg", "BayesAvg"),
                       ("cover_group", "CoverGroup"), ("cover_unit", "CoverUnit"), ("tau", "Tau"), ("omega", "Omega")):
            macro(f"Eeight{tag}", f"{np.mean([x[k] for x in res]):.2f}")
        macro("EeightReps", f"{len(res)}")


# ------------------------------------------------------------------------------------------------------
# Figure 4: pilot illustration (Vermont candidate places; not adjudicated)
# ------------------------------------------------------------------------------------------------------
def figure_pilot(path):
    import pandas as pd
    pm = pd.read_csv(path)
    stars = ["n_1star", "n_2star", "n_3star", "n_4star", "n_5star"]
    pl = pm.groupby(["gmap_id", "service_family", "name", "address"], as_index=False)[stars].sum()
    name = pl["name"].str.strip()
    cls = np.where(name.eq("United States Postal Service"), "Post offices",
                   np.where(name.str.startswith("Vermont DMV"), "DMV offices",
                            np.where((pl.service_family == "public_library") & name.str.contains("Library"), "Public libraries", "")))
    pl = pl.assign(cls=cls)[cls != ""].reset_index(drop=True)
    classes = ["DMV offices", "Post offices", "Public libraries"]
    X = np.zeros((len(pl), 3, 5))
    for i, c in enumerate(classes):
        m = (pl.cls == c).values
        X[m, i, :] = pl.loc[m, stars].values
    fit = RatingScaleEB().fit(X)
    N = X.sum((1, 2)); raw = (X.sum(1) @ NU) / N
    lo, hi = fit.interval()
    pl = pl.assign(N=N, raw=raw, theta=fit.theta_, sd=fit.theta_sd_, lo=lo, hi=hi)
    LAM = 1.5
    dl, dh = [], []
    for j in range(len(pl)):
        psi = fit.psi_[classes.index(pl.cls[j])]
        dl.append(drift_interval(lo[j], psi, NU, LAM)[0]); dh.append(drift_interval(hi[j], psi, NU, LAM)[1])
    pl = pl.assign(dlo=dl, dhi=dh)
    pl.drop(columns=["address"]).to_csv(OUT / "pilot-index.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), gridspec_kw=dict(wspace=0.32, width_ratios=[1.25, 1]))
    ax = axes[0]
    d = pl[pl.cls == "Post offices"].sort_values("theta").reset_index(drop=True)
    y = np.arange(len(d))
    ax.hlines(y, d.dlo, d.dhi, color="#b9cfee", lw=2.2, zorder=1)
    ax.hlines(y, d.lo, d.hi, color=BLUE, lw=0.9, zorder=2)
    ax.scatter(d.theta, y, s=9 + 5 * np.log(d.N), color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color=INK, lw=0.6); ax.set_yticks([]); ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"Index $\theta_j$ (log odds per star, relative to class mean)"); ax.set_ylabel("Post offices, ordered by index")
    panel(ax, "(a) Post offices: index and intervals")
    ax = axes[1]
    for c, col, mk in zip(classes, (ORANGE, BLUE, AQUA), ("s", "o", "^")):
        m = (pl.cls == c).values
        ax.scatter(pl.raw[m], pl.theta[m], s=8 + 6 * np.log(pl.N[m]), color=col, marker=mk, edgecolor="white", linewidth=0.5, label=c, alpha=0.9)
    ax.set_xlabel("Mean stars"); ax.set_ylabel(r"Index $\theta_j$"); ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    panel(ax, "(b) Shrinkage of sparse units")
    fig.savefig(FIG / "fig4-pilot.pdf"); plt.close(fig)

    lines = [r"\begin{tabular}{@{}ll ccccc@{}}\toprule", r"Class & Units & 1 & 2 & 3 & 4 & 5 stars\\\midrule"]
    for i, c in enumerate(classes):
        m = np.where((pl.cls == c).values)[0]
        med = np.median(pl.theta[m])
        for half, idx in (("below median index", m[pl.theta[m] <= med]), ("above median index", m[pl.theta[m] > med])):
            obs = X[idx, i, :].sum(0)
            ex = family_probs(np.array([_invert_mean((obs @ NU) / obs.sum(), fit.psi_[i], NU)]), fit.psi_[i], NU)[0]
            lines.append(f"{c} & {half}, observed & " + " & ".join(f"{v:.2f}" for v in obs / obs.sum()) + r"\\")
            lines.append(f" & {half}, fitted & " + " & ".join(f"{v:.2f}" for v in ex / ex.sum()) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (TAB / "tabS3-pilotfit.tex").write_text("\n".join(lines) + "\n")
    macro("PilotUnits", f"{len(pl)}"); macro("PilotRatings", f"{int(N.sum()):,}".replace(",", "{,}"))
    macro("PilotMedianN", f"{np.median(N):.0f}"); macro("PilotShareLtFive", f"{100 * np.mean(N < 5):.0f}")
    macro("PilotTau", f"{fit.tau_:.2f}")
    for c, tag in zip(classes, ("DMV", "USPS", "Lib")):
        m = (pl.cls == c).values
        macro(f"Pilot{tag}Units", f"{m.sum()}"); macro(f"Pilot{tag}Ratings", f"{int(N[m].sum())}")
    # a worked comparison: the two DMV offices with the most ratings among the top and bottom of the class
    dmv = pl[pl.cls == "DMV offices"].sort_values("theta")
    a, bb = dmv.iloc[-1], dmv.iloc[0]
    psi = fit.psi_[0]
    macro("PilotDMVHigh", a["name"].replace("Vermont DMV - ", "")); macro("PilotDMVLow", bb["name"].replace("Vermont DMV - ", ""))
    macro("PilotDMVHighStars", f"{a.raw:.2f}"); macro("PilotDMVLowStars", f"{bb.raw:.2f}")
    macro("PilotDMVDiff", f"{a.theta - bb.theta:.2f}")
    macro("PilotDMVDiffSE", f"{np.hypot(a.sd, bb.sd):.2f}")
    macro("PilotDMVBreakdownPoint", f"{breakdown(a.theta, bb.theta, psi, NU):.1f}")   # at the point estimates
    macro("PilotDMVBreakdown", f"{breakdown(a.lo, bb.hi, psi, NU):.1f}")              # sampling-adjusted (95 percent intervals)
    five = pl[(pl.raw == 5.0)]
    macro("PilotAllFive", f"{len(five)}"); macro("PilotAllFiveMaxN", f"{int(five.N.max()) if len(five) else 0}")
    usps = pl[pl.cls == "Post offices"]
    macro("PilotUSPSRobust", f"{int(((usps.dlo > 0) | (usps.dhi < 0)).sum())}")
    macro("PilotUSPSSignif", f"{int(((usps.lo > 0) | (usps.hi < 0)).sum())}")
    macro("PilotUSPSMedianN", f"{np.median(usps.N):.0f}")
    macro("PilotLambda", f"{LAM:g}")


if __name__ == "__main__":
    figure_identification()
    theory_numbers()
    figure_invariance()
    figure_sparse()
    table_sources_drift()
    tables_supplement()
    if "--pilot" in sys.argv:
        figure_pilot(sys.argv[sys.argv.index("--pilot") + 1])
    (TAB / "macros.tex").write_text("% Generated by code/make_exhibits.py; do not edit.\n" +
                                   "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n")
    print(f"wrote {len(MACROS)} macros")
