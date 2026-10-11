"""Turn simulation output into the manuscript's figures (vector PDF), tables and number macros.

Run from code/:   python3 make_exhibits.py
Reads ../results/simulations/E*.json, writes ../results/figures/*.pdf and ../results/tables/*.tex.
No number in the manuscript is typed by hand: each one is a macro defined in ../results/tables/macros.tex.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

from vrindex import (RatingScaleEB, family_probs, _invert_mean, drift_interval, breakdown, contamination_interval,
                     simulate_class, hilbert_distance)

OUT, FIG, TAB = Path("../results/simulations"), Path("../results/figures"), Path("../results/tables")
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
NU = np.arange(1, 6.0)
C_LATENT = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35]))
B_NEG, B_J = np.array([6, 3, 1.5, 1, 1.0]), np.array([15, 2, 1, 1.5, 5.0])

# ---- figure style: sans-serif (Arial metrics), thin marks, recessive axes, validated three-hue palette ---------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, PALE = "#0b0b0b", "#6b6a66", "#e4e3df", "#b9cfee"
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
METHODS = [("raw", "Mean stars"), ("fav", "Share of 4-5 stars"), ("wilson", "Wilson lower bound"), ("bayes", "Bayesian average"),
           ("cumulative", "Hierarchical ordered logit"), ("index", "Index (this article)")]


def macro(name, value):
    MACROS[name] = value


def panel(ax, tag):
    ax.set_title(tag, loc="left", fontweight="bold", pad=4)


def load(name):
    p = OUT / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def mean_of(res, *keys):
    def get(x):
        for k in keys:
            x = x[k]
        return x
    return float(np.mean([get(x) for x in res]))


def mcse_of(res, *keys):
    def get(x):
        for k in keys:
            x = x[k]
        return x
    v = np.array([get(x) for x in res], float)
    return float(v.std(ddof=1) / np.sqrt(len(v)))


# ------------------------------------------------------------------------------------------------------
# Figure 2 and Table B1: the identification argument in one example (population quantities, no sampling)
# ------------------------------------------------------------------------------------------------------
def figure_identification():
    theta = np.array([-0.5, 0.0, 0.5])
    pi = family_probs(theta, C_LATENT, NU)
    vis = pi * B_J; vis /= vis.sum(1, keepdims=True)
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
    panel(ax, "(c) Log ratios between units")
    fig.savefig(FIG / "fig_identification.pdf"); plt.close(fig)
    macro("ExLatentMeans", ", ".join(f"{m:.2f}" for m in pi @ NU))
    macro("ExVisibleMeans", ", ".join(f"{m:.2f}" for m in vis @ NU))
    macro("ExLatentB", f"{(pi @ NU)[1]:.2f}"); macro("ExLatentC", f"{(pi @ NU)[2]:.2f}")

    # the same three units under four visibility functions
    from scipy.optimize import minimize
    from scipy.special import expit

    def cumlogit(P):
        def nll(par):
            kap = np.cumsum(np.concatenate([[par[0]], np.exp(par[1:4])])); loc = np.concatenate([[0.0], par[4:]])
            cdf = expit(kap[None, :] - loc[:, None]); cdf = np.concatenate([np.zeros((3, 1)), cdf, np.ones((3, 1))], 1)
            return -(P * np.log(np.clip(np.diff(cdf, axis=1), 1e-300, None))).sum()
        r = minimize(nll, np.array([-2, 0, 0, 0, 0, 0.0]), method="BFGS"); return np.concatenate([[0.0], r.x[4:]])
    voices = [("None (all encounters visible)", np.ones(5)), ("Negativity", B_NEG), ("J-shaped", B_J),
              ("J-shaped, moderated", B_J * np.array([0.5, 0.7, 1, 1, 1.0]))]
    rows = []
    for name, bb in voices:
        v = pi * bb; v /= v.sum(1, keepdims=True)
        psi = np.log(v[1]); psi -= psi[0]
        th = np.array([_invert_mean(x @ NU, psi, NU) for x in v])
        rows.append((name, np.diff(v @ NU), np.diff(v[:, 3:].sum(1)), np.diff(cumlogit(v)), np.diff(th)))
    lines = [r"\begin{tabular}{@{}l cc cc cc cc@{}}\toprule",
             r" & \multicolumn{2}{c}{Mean stars} & \multicolumn{2}{c}{Share 4--5 stars} & \multicolumn{2}{c}{Ordered logit} & \multicolumn{2}{c}{Index $\theta$}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
             r"Visibility function & B$-$A & C$-$B & B$-$A & C$-$B & B$-$A & C$-$B & B$-$A & C$-$B\\\midrule"]
    for name, a1, a2, a3, a4 in rows:
        lines.append(f"{name} & " + " & ".join(f"{x:.2f}" for pair in (a1, a2, a3, a4) for x in pair) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (TAB / "tab_example.tex").write_text("\n".join(lines) + "\n")
    macro("ExStarsNone", f"{rows[0][1][0]:.2f}"); macro("ExStarsJ", f"{rows[2][1][0]:.2f}")


# ------------------------------------------------------------------------------------------------------
# Figure 3: what each assumption identifies (population quantities for units B and C of Figure 2)
# ------------------------------------------------------------------------------------------------------
def figure_ladder():
    pi = family_probs(np.array([0.0, 0.5]), C_LATENT, NU)
    q = pi * B_J; q /= q.sum(1, keepdims=True)
    qB, qC = q
    mu_true = float(pi[1] @ NU)
    lor = lambda a, b_: float(np.log(a[4] / a[0]) - np.log(b_[4] / b_[0]))
    w_true = lor(pi[1], pi[0])                                        # = (nu_K - nu_1) * 0.5 = 2

    def level_bounds(Lam):                                            # b known, departure up to Lam in unit C
        base = qC / B_J
        vals = []
        for k in range(6):
            for up in (True, False):
                v = np.ones(5)
                if up:
                    v[k:] = Lam
                else:
                    v[:k] = Lam
                p = base * v; vals.append((p / p.sum()) @ NU)
        return min(vals), max(vals)
    eps = 0.10
    w_fab = float(np.log((qC[4] - eps) / qC[0]) - np.log(qB[4] / qB[0]))
    rows = [  # label, level set for unit C, log-odds-ratio set   (None = unbounded on that side)
        ("No restriction on visibility", (1, 5), (None, None)),
        (r"Departures up to $\Lambda=2$ in both units", (1, 5), (w_true - 2 * np.log(2), w_true + 2 * np.log(2))),
        (r"Departures up to $\Lambda=1.5$ in both units", (1, 5), (w_true - 2 * np.log(1.5), w_true + 2 * np.log(1.5))),
        (r"Favorable departure up to $\Lambda=2$, unit C only", (1, 5), (w_true - np.log(2), w_true)),
        (r"Up to 10% fabricated five-star ratings, unit C", (1, 5), (w_fab, w_true)),
        ("Shared visibility (Assumption 1)", (1, 5), (w_true, w_true)),
        (r"Shared visibility, anchor, departure up to $\Lambda=1.5$ in C", level_bounds(1.5), (w_true - np.log(1.5), w_true + np.log(1.5))),
        ("Shared visibility and an anchor", (mu_true, mu_true), (w_true, w_true)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08, width_ratios=[1, 1.15]))
    y = np.arange(len(rows))[::-1]
    for ax, col, truth, xlim, xl, ttl in ((axes[0], 1, mu_true, (0.8, 5.2), "Mean evaluation of unit C, all encounters (stars)", "(a) A level"),
                                          (axes[1], 2, w_true, (-0.6, 4.6), "Log odds ratio, five stars vs. one star, C vs. B", "(b) A comparison")):
        ax.axvline(truth, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
        for yy, r in zip(y, rows):
            lo, hi = r[col]
            if lo is None:                                           # unbounded: draw to the frame with arrowheads
                ax.annotate("", xy=(xlim[0] + 0.02, yy), xytext=(xlim[1] - 0.02, yy),
                            arrowprops=dict(arrowstyle="<|-|>", color=PALE, lw=2.4, mutation_scale=9), zorder=2)
            elif hi - lo < 1e-9:
                ax.plot([lo], [yy], marker="o", color=BLUE, mec="white", mew=0.8, ms=6, zorder=3)
            else:
                full = col == 1 and abs(lo - 1) < 1e-9 and abs(hi - 5) < 1e-9
                ax.plot([lo, hi], [yy, yy], color=PALE if full else BLUE, lw=4.2 if full else 3.0, solid_capstyle="butt", zorder=2)
        ax.set_xlim(*xlim); ax.set_xlabel(xl); panel(ax, ttl); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows], color=INK); axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.savefig(FIG / "fig_ladder.pdf"); plt.close(fig)
    macro("LadderOmega", f"{w_true:.1f}"); macro("LadderMu", f"{mu_true:.2f}")
    lo, hi = level_bounds(1.5); macro("LadderLevelLo", f"{lo:.2f}"); macro("LadderLevelHi", f"{hi:.2f}")
    macro("LadderFabLo", f"{w_fab:.2f}")


# ------------------------------------------------------------------------------------------------------
# Population-level numbers quoted in the text (sensitivity analysis, counterexamples); no sampling involved
# ------------------------------------------------------------------------------------------------------
def theory_numbers():
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp
    psi = C_LATENT + np.log(B_J); psi -= psi[0]

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
    macro("BreakQuarterOne", f"{breakdown(0.125, -0.125, psi, NU, suspect='high', direction='favorable'):.1f}")
    macro("BreakHalfOne", f"{breakdown(0.25, -0.25, psi, NU, suspect='high', direction='favorable'):.1f}")
    macro("BreakQuarterBound", f"{np.exp(0.5):.2f}"); macro("BreakHalfBound", f"{np.exp(1.0):.2f}")
    # fabricated ratings: mean-based bounds versus the set that uses the whole histogram
    mlo, mhi = contamination_interval(0.0, psi, NU, 0.2)
    grid = np.linspace(-3, 3, 60001); P = family_probs(grid, psi, NU); q = family_probs(np.array(0.0), psi, NU)
    ok = grid[((1 - 0.2) * P <= q[None, :] + 1e-15).all(1)]
    macro("ContamMean", f"[{mlo:.2f}, {mhi:.2f}]".replace("-", "$-$")); macro("ContamHist", f"[{ok.min():.2f}, {ok.max():.2f}]".replace("-", "$-$"))
    # rater types: Hilbert diameter of the two type-specific visibility functions, and the drift of a 30/70 against a 70/30 unit
    macro("TypeBound", f"{np.exp(hilbert_distance(B_NEG, B_J)):.1f}")
    macro("TypeModerate", f"{np.exp(hilbert_distance(0.7 * B_NEG + 0.3 * B_J, 0.3 * B_NEG + 0.7 * B_J)):.1f}")
    # a latent law that is exactly cumulative-logit: fitted location differences under two visibility functions
    kap0 = np.array([-2.9, -1.9, -0.9, 0.6]); loc0 = np.array([0.0, 0.5, 1.0])
    cdf0 = expit(kap0[None, :] - loc0[:, None]); pi_cl = np.diff(np.concatenate([np.zeros((3, 1)), cdf0, np.ones((3, 1))], 1), axis=1)

    def cumlogit(Pm):
        def nll(par):
            kap = np.cumsum(np.concatenate([[par[0]], np.exp(par[1:4])])); loc = np.concatenate([[0.0], par[4:]])
            cdf = expit(kap[None, :] - loc[:, None]); cdf = np.concatenate([np.zeros((3, 1)), cdf, np.ones((3, 1))], 1)
            return -(Pm * np.log(np.clip(np.diff(cdf, axis=1), 1e-300, None))).sum()
        r = minimize(nll, np.array([-2, 0, 0, 0, 0, 0.0]), method="BFGS"); return np.diff(np.concatenate([[0.0], r.x[4:]]))
    for tag, bb in (("Neg", B_NEG), ("J", B_J)):
        v = pi_cl * bb; v /= v.sum(1, keepdims=True); d = cumlogit(v)
        macro(f"CumExact{tag}", f"{d[0]:.2f} and {d[1]:.2f}")
    # polarisation example (Appendix E)
    quad = (NU - NU.mean()) ** 2
    for tag, (th, ph) in (("Moderate", (0.15, -0.35)), ("Polar", (0.0, 0.35))):
        e = C_LATENT + th * NU + ph * quad; pl_ = np.exp(e - logsumexp(e)); v = pl_ * B_J; v /= v.sum()
        macro(f"PolLat{tag}", f"{pl_ @ NU:.2f}"); macro(f"PolVis{tag}", f"{v @ NU:.2f}")
    # index from a rating count and a rounded average (Appendix B)
    d = simulate_class(1500, np.random.default_rng(77), b=B_J)
    X = d["X"]; X = X[X.sum(1) > 0].astype(float)
    fit = RatingScaleEB().fit(X)
    N = X.sum(1); avg = np.round((X @ NU) / N, 1)
    post = fit._posterior(np.atleast_2d(fit.psi_), np.zeros(len(N)), fit.tau_, X[:, None, :], N[:, None], (N * avg)[:, None])
    macro("RoundCorr", f"{np.corrcoef(post['mean'], fit.theta_)[0, 1]:.3f}")
    macro("RoundMaxDev", f"{np.max(np.abs(post['mean'] - fit.theta_)):.2f}")
    inner = (avg > 1.05) & (avg < 4.95)
    macro("RoundMaxDevInner", f"{np.max(np.abs(post['mean'] - fit.theta_)[inner]):.2f}")


# ------------------------------------------------------------------------------------------------------
# Figure 4: comparison with existing summaries (E1, E2, E3)
# ------------------------------------------------------------------------------------------------------
def figure_scorecard():
    r1, r2, r3 = load("E1"), load("E2"), load("E3")
    if r1 is None or r2 is None or r3 is None:
        return
    labels = [m[1] for m in METHODS]; keys = [m[0] for m in METHODS]
    y = np.arange(len(keys))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), sharey=True, gridspec_kw=dict(wspace=0.10, hspace=0.55))
    axes = axes.ravel()
    colr = lambda k: BLUE if k == "index" else MUTED

    # (a) stability of an effect estimate when only the visibility function changes
    ax = axes[0]
    regimes = [("negativity", "^", "Negativity"), ("J-shaped", "o", "J-shaped"), ("J-shaped, moderated", "s", "J-shaped, moderated")]
    res = r1["results"]
    for k, yy in zip(keys, y):
        base = np.mean([x["none"][k] for x in res])
        for (g, mk, _), off in zip(regimes, (0.22, 0.0, -0.22)):
            ratio = np.mean([x[g][k] for x in res]) / base
            ax.plot([ratio], [yy + off], marker=mk, color=colr(k), mec="white", mew=0.7, ms=5.5, ls="none")
            tag = {"negativity": "Neg", "J-shaped": "J", "J-shaped, moderated": "Mod"}[g]
            macro(f"Ratio{k.capitalize()}{tag}", f"{ratio:.2f}")
    ax.axvline(1, color=INK, lw=0.6)
    for (g, mk, lab) in regimes:
        ax.plot([], [], marker=mk, color=MUTED, ls="none", mec="white", mew=0.7, label=lab)
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.25)
    ax.set_xlabel("Effect estimate relative to its value\nwhen every encounter is visible"); panel(ax, "(a) Stability of scale")
    ax.set_xlim(0.85, 2.3)

    # (b), (c) sparse data
    res2 = r2["results"]
    for ax, field, xl, ttl, tagname in ((axes[1], "overall", "Rank correlation with the truth", "(b) Ranking under sparse data", "Rho"),
                                        (axes[2], "bottom_decile_precision", "Share of units flagged as worst decile\nthat belong there", "(c) Finding the worst units", "Bottom")):
        for k, yy in zip(keys, y):
            v = np.mean([x[field][k] for x in res2])
            ax.plot([v], [yy], marker="o", color=colr(k), mec="white", mew=0.7, ms=6, ls="none")
            ax.plot([0, v], [yy, yy], color=colr(k), lw=0.8, alpha=0.5)
            macro(f"Etwo{tagname}{k.capitalize()}", f"{v:.2f}")
        ax.set_xlabel(xl); panel(ax, ttl); ax.set_xlim(0.35, 0.95)

    # (d) two sources, mix related to the truth
    ax = axes[3]
    res3 = [x for x, t in zip(r3["results"], r3["tasks"]) if abs(t[2] - 0.5) < 1e-9]
    for k, yy in zip(keys, y):
        kk = "index_stratified" if k == "index" else k
        v = np.mean([x[kk] for x in res3])
        ax.plot([v], [yy], marker="o", color=colr(k), mec="white", mew=0.7, ms=6, ls="none")
        ax.plot([0, v], [yy, yy], color=colr(k), lw=0.8, alpha=0.5)
    vd = np.mean([x["demeaned"] for x in res3])
    ax.plot([vd], [y[-1] + 0.5], marker="D", color=AQUA, mec="white", mew=0.7, ms=5.5, ls="none")
    ax.text(vd - 0.02, y[-1] + 0.5, "source-demeaned\naverage", ha="right", va="center", fontsize=6.8, color=INK)
    ax.set_xlabel("Rank correlation with the truth"); panel(ax, "(d) Two sources, mix related to the truth"); ax.set_xlim(0.35, 0.95)
    for ax in (axes[0], axes[2]):
        ax.set_yticks(y); ax.set_yticklabels(labels, color=INK); ax.tick_params(axis="y", length=0)
    for ax in axes:
        ax.grid(axis="y", visible=False); ax.set_ylim(-0.6, len(keys) - 0.4)
    fig.savefig(FIG / "fig_scorecard.pdf"); plt.close(fig)

    # largest Monte Carlo standard error among the quantities plotted
    mc = 0.0
    for k in keys:
        base = np.mean([x["none"][k] for x in res])
        for g, _, _ in regimes:
            mc = max(mc, np.std([x[g][k] for x in res], ddof=1) / np.sqrt(len(res)) / abs(base))
        for field in ("overall", "bottom_decile_precision"):
            mc = max(mc, np.std([x[field][k] for x in res2], ddof=1) / np.sqrt(len(res2)))
        kk = "index_stratified" if k == "index" else k
        mc = max(mc, np.std([x[kk] for x in res3], ddof=1) / np.sqrt(len(res3)))
    macro("ScorecardMCSE", f"{np.ceil(mc * 1000) / 1000:.3f}")
    # macros for the text
    macro("EoneReps", f"{len(res)}"); macro("EoneJ", f"{r1['J']:,}".replace(",", "{,}"))
    for g, tag in (("none", "None"), ("negativity", "Neg"), ("J-shaped", "J"), ("J-shaped, moderated", "Mod")):
        macro(f"EoneIndex{tag}", f"{np.mean([x[g]['index'] for x in res]):.3f}")
        macro(f"EoneRaw{tag}", f"{np.mean([x[g]['raw'] for x in res]):.2f}")
    latent = np.mean([x["none"]["latent_stars"] for x in res]); macro("EoneLatentStars", f"{latent:.2f}")
    macro("EoneIndexDev", f"{100 * max(abs(np.mean([x[g]['index'] for x in res]) / 0.30 - 1) for g in res[0]):.0f}")
    macro("EoneOverstate", f"{100 * (max(np.mean([x[g]['raw'] for x in res]) for g in res[0]) / latent - 1):.0f}")
    macro("EoneConverged", f"{100 * np.mean([all(x[g]['converged'] for g in x) for x in res]):.0f}")
    macro("EtwoCoverage", f"{np.mean([x['coverage'] for x in res2]):.3f}")
    macro("EtwoShareLtFive", f"{100 * np.mean([x['share_lt5'] for x in res2]):.0f}")
    macro("EtwoMedianN", f"{np.median([x['median_n'] for x in res2]):.0f}")
    macro("EtwoReps", f"{len(res2)}"); macro("EtwoConverged", f"{100 * np.mean([x['converged'] for x in res2]):.0f}")
    macro("EthreeReps", f"{int(len(r3['results']) / 2)}")
    macro("EthreeDemeanedHalfText", f"{vd:.2f}")


# ------------------------------------------------------------------------------------------------------
# Figure 5: unit-specific curation (E9) and fabrication (E10)
# ------------------------------------------------------------------------------------------------------
def figure_manipulation():
    r9, r10 = load("E9"), load("E10")
    if r9 is None or r10 is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.5), gridspec_kw=dict(wspace=0.32, hspace=0.62))
    meth = [("raw", "Mean stars", ORANGE, "s"), ("bayes", "Bayesian average", AQUA, "^"), ("wilson", "Wilson lower bound", MUTED, "D"),
            ("index", "Index (this article)", BLUE, "o")]
    for row, (r, key, xl, tag) in enumerate(((r9, "kappa", "Share of one- and two-star ratings removed\nin manipulating units", "Nine"),
                                             (r10, "eps", "Fabricated five-star ratings as a share of\nvisible ratings in manipulating units", "Ten"))):
        res = r["results"]; xs = sorted(set(x[key] for x in res))
        ax = axes[row, 0]
        for k, lab, col, mk in meth:
            v = [np.mean([x["rank_gain"][k] for x in res if x[key] == s]) for s in xs]
            ax.plot(xs, v, color=col, marker=mk, mec="white", mew=0.8, label=lab, lw=1.3, zorder=3 if k == "index" else 2)
        ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(xs); ax.set_xlabel(xl); ax.set_ylabel("Percentile ranks gained")
        panel(ax, "(a) Curation: what manipulators gain" if row == 0 else "(c) Fabrication: what manipulators gain")
        if row == 0:
            ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.1)
        ax = axes[row, 1]
        series = [("cov_plain", "Plain 95% interval", MUTED, "s", "--"), ("cov_directional", "One-sided sensitivity interval", BLUE, "o", "-")]
        for f, lab, col, mk, ls in series:
            v = [np.mean([x[f] for x in res if x[key] == s]) for s in xs]
            ax.plot(xs, v, color=col, marker=mk, ls=ls, mec="white", mew=0.8, label=lab, lw=1.3)
        v = [np.mean([x["cov_clean"] for x in res if x[key] == s]) for s in xs]
        ax.plot(xs, v, color=AQUA, marker="^", mec="white", mew=0.8, lw=1.0, label="Plain interval, other units")
        ax.axhline(0.95, color=INK, lw=0.6); ax.set_ylim(0.5, 1.01); ax.set_xticks(xs); ax.set_xlabel(xl); ax.set_ylabel("Coverage of the truth")
        panel(ax, "(b) Curation: coverage" if row == 0 else "(d) Fabrication: coverage")
        if row == 0:
            ax.legend(loc="lower left", handlelength=1.8, borderaxespad=0.1)
        for s in xs:
            sel = [x for x in res if x[key] == s]
            t = f"E{tag}" + {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQ", 0.1: "OneTenth", 0.2: "OneFifth", 0.3: "ThreeTenths"}[s]
            macro(t + "GainIndex", f"{int(round(np.mean([x['rank_gain']['index'] for x in sel])))}")
            macro(t + "GainRaw", f"{int(round(np.mean([x['rank_gain']['raw'] for x in sel])))}")
            macro(t + "GainWilson", f"{int(round(np.mean([x['rank_gain']['wilson'] for x in sel])))}")
            macro(t + "GainBayes", f"{int(round(np.mean([x['rank_gain']['bayes'] for x in sel])))}")
            if "share_unbounded" in sel[0]:
                macro(t + "Unbounded", f"{100 * np.mean([x['share_unbounded'] for x in sel]):.0f}")
            macro(t + "CovPlain", f"{np.mean([x['cov_plain'] for x in sel]):.2f}")
            macro(t + "CovDir", f"{np.mean([x['cov_directional'] for x in sel]):.2f}")
            macro(t + "WidthPlain", f"{np.mean([x['width_plain'] for x in sel]):.2f}")
            macro(t + "WidthDir", f"{np.mean([x['width_directional'] for x in sel]):.2f}")
            macro(t + "Bias", f"{np.mean([x['bias'] for x in sel]):.2f}")
        macro(f"E{tag}Reps", f"{int(len(res) / len(xs))}")
        macro(f"E{tag}CovClean", f"{np.mean([x['cov_clean'] for x in res]):.2f}")
    fig.savefig(FIG / "fig_manipulation.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------------
# Figure 6: sparse data (E2) and the price of robustness (computed here from one simulated class per volume)
# ------------------------------------------------------------------------------------------------------
def figure_sparse_robustness():
    r = load("E2")
    if r is None:
        return
    res = r["results"]
    bins = ["1-2", "3-5", "6-20", "21-100", "101-inf"]; labs = ["1-2", "3-5", "6-20", "21-100", ">100"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.42))
    rm = [np.mean([x["bins"][b]["rmse_index"] for x in res if b in x["bins"]]) for b in bins]
    cv = [np.mean([x["bins"][b]["coverage"] for x in res if b in x["bins"]]) for b in bins]
    sh = [np.mean([x["bins"][b]["share"] for x in res if b in x["bins"]]) for b in bins]
    ax = axes[0]
    ax.plot(range(5), rm, color=BLUE, marker="o", mec="white", mew=0.8)
    ax.axhline(0.45, color=INK, lw=0.6); ax.text(4.3, 0.462, "no ratings", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n({100 * s:.0f}%)" for l, s in zip(labs, sh)]); ax.set_ylim(0, 0.5)
    ax.set_xlabel("Ratings per unit (share of units)"); ax.set_ylabel(r"Root mean squared error of $\hat\theta_j$"); panel(ax, "(a) Accuracy")
    ax = axes[1]
    ax.plot(range(5), cv, color=BLUE, marker="o", mec="white", mew=0.8); ax.axhline(0.95, color=INK, lw=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.set_ylim(0.85, 1.0)
    ax.set_xlabel("Ratings per unit"); ax.set_ylabel("Coverage of 95% intervals"); panel(ax, "(b) Calibration")
    for b_, v_ in zip("ABCDE", rm):
        macro(f"EtwoRmse{b_}", f"{v_:.2f}")
    # (c) share of pairs that can still be ordered when departures up to Lambda are allowed in both units
    ax = axes[2]
    lams = np.array([1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    n_class, mc = 8, 0.0
    for med, col, mk in ((6, MUTED, "s"), (20, AQUA, "^"), (100, BLUE, "o")):
        shares = []
        for rep in range(n_class):
            rng = np.random.default_rng([20260921, 77, med, rep])
            d = simulate_class(2000, rng, c=C_LATENT, b=B_J, tau=0.45, med_n=float(med))
            X = d["X"].astype(float); X = X[X.sum(1) > 0]
            rs = RatingScaleEB().fit(X); lo, hi = rs.interval()
            sub = rng.choice(len(lo), 400, replace=False)
            row = []
            for L in lams:
                Lo = np.array([drift_interval(lo[j], rs.psi_, NU, L)[0] for j in sub]); Hi = np.array([drift_interval(hi[j], rs.psi_, NU, L)[1] for j in sub])
                row.append(np.mean((Lo[:200] > Hi[200:]) | (Lo[200:] > Hi[:200])))
            shares.append(row)
        shares = np.array(shares); share = shares.mean(0); mc = max(mc, (shares.std(0, ddof=1) / np.sqrt(n_class)).max())
        ax.plot(lams, share, color=col, marker=mk, mec="white", mew=0.8, lw=1.3, label=f"median {med} ratings")
        word = {6: "Six", 20: "Twenty", 100: "Hundred"}[med]
        macro(f"Robust{word}One", f"{100 * share[0]:.0f}"); macro(f"Robust{word}Two", f"{100 * share[3]:.0f}")
    macro("RobustMCSE", f"{100 * mc:.0f}"); macro("RobustClasses", f"{n_class}")
    ax.set_xlabel(r"Departure allowed in both units, $\Lambda$"); ax.set_ylabel("Share of pairs ordered"); ax.set_ylim(0, None)
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.0); panel(ax, "(c) The price of robustness")
    fig.savefig(FIG / "fig_sparse_robustness.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------------
# Appendix C exhibits
# ------------------------------------------------------------------------------------------------------
def figure_invariance():
    r = load("E1")
    if r is None:
        return
    regimes = ["none", "negativity", "J-shaped", "J-shaped, moderated"]
    est = {"raw": ("Mean stars", MUTED, "s", "-"), "bayes": ("Bayesian average", MUTED, "^", "--"), "fav": ("Share of 4-5 stars", AQUA, "v", "-"),
           "cumulative": ("Hierarchical ordered logit", ORANGE, "D", "-"), "index": ("Index (this article)", BLUE, "o", "-")}
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    xs = np.arange(len(regimes))
    for i, (k, (lab, col, mk, ls)) in enumerate(est.items()):
        A = np.array([[x[g][k] for g in regimes] for x in r["results"]])
        ratio = A / A[:, [0]].mean(); m = ratio.mean(0); lo, hi = np.percentile(ratio, [2.5, 97.5], axis=0)
        off = (i - 2) * 0.07
        ax.errorbar(xs + off, m, yerr=[m - lo, hi - m], color=col, marker=mk, ls=ls, mec="white", mew=0.8, capsize=0, elinewidth=1.0,
                    label=lab, zorder=3 if k == "index" else 2)
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(["None", "Negativity", "J-shaped", "J-shaped,\nmoderated"]); ax.set_xlim(-0.45, len(regimes) - 0.55)
    ax.set_xlabel("Visibility function"); ax.set_ylabel("Estimated effect relative to\nits value without selection")
    ax.legend(loc="upper left", handlelength=2.2, ncol=1)
    fig.savefig(FIG / "fig_invariance.pdf"); plt.close(fig)


def table_sources():
    r3 = load("E3")
    if r3 is None:
        return
    tasks = np.array(r3["tasks"]); res = r3["results"]
    names = [("raw", "Pooled mean stars"), ("fav", "Pooled share of 4--5 stars"), ("wilson", "Pooled Wilson lower bound"),
             ("bayes", "Pooled Bayesian average"), ("cumulative", "Pooled hierarchical ordered logit"),
             ("demeaned", "Source-demeaned average"), ("index_pooled", "Index, sources pooled"), ("index_stratified", "Index, stratified by source")]
    lines = [r"\begin{tabular}{@{}lcc@{}}\toprule", r" & \multicolumn{2}{c}{Source mix}\\\cmidrule(l){2-3}",
             r"Summary & unrelated to $\theta_j$ & related to $\theta_j$\\\midrule"]
    mc = 0.0
    for k, lab in names:
        vals = []
        for c in (0.0, 0.5):
            v = np.array([x[k] for x, t in zip(res, tasks) if abs(t[2] - c) < 1e-9]); vals.append(v.mean()); mc = max(mc, v.std(ddof=1) / np.sqrt(len(v)))
        lines.append(f"{lab} & {vals[0]:.2f} & {vals[1]:.2f}" + r"\\")
        macro(f"Ethree{k.replace('_', '').capitalize()}Zero", f"{vals[0]:.2f}"); macro(f"Ethree{k.replace('_', '').capitalize()}Half", f"{vals[1]:.2f}")
    lines.append(r"\bottomrule\end{tabular}")
    (TAB / "tab_sources.tex").write_text("\n".join(lines) + "\n")
    macro("EthreeMCSE", f"{np.ceil(mc * 1000) / 1000:.3f}")


def table_drift():
    r4 = load("E4")
    if r4 is None:
        return
    tasks = np.array(r4["tasks"]); res = r4["results"]
    lines = [r"\begin{tabular}{@{}l cccc c cccc@{}}\toprule",
             r" & \multicolumn{4}{c}{Coverage of $\theta_j$} & & \multicolumn{4}{c}{Share of pairs ordered}\\\cmidrule(lr){2-5}\cmidrule(l){7-10}",
             r"Departure present, $\Lambda_0$ & $\Lambda{=}1$ & $1.5$ & $2$ & $3$ & & $\Lambda{=}1$ & $1.5$ & $2$ & $3$\\\midrule"]
    worst, mc = 0.0, 0.0
    for L0 in (1.0, 1.5, 2.0):
        sel = [x for x, t in zip(res, tasks) if abs(t[2] - L0) < 1e-9]
        cov = [np.mean([x[str(L)]["coverage"] for x in sel]) for L in (1.0, 1.5, 2.0, 3.0)]
        dec = [np.mean([x[str(L)]["declared"] for x in sel]) for L in (1.0, 1.5, 2.0, 3.0)]
        worst = max(worst, max(np.mean([x[str(L)]["wrong_among_declared"] for x in sel]) for L in (1.0, 1.5, 2.0, 3.0) if L >= L0))
        for L in ("1.0", "1.5", "2.0", "3.0"):
            for kk in ("coverage", "declared"):
                mc = max(mc, np.std([x[L][kk] for x in sel], ddof=1) / np.sqrt(len(sel)))
        lines.append(f"{L0:g} & " + " & ".join(f"{v:.2f}" for v in cov) + " & & " + " & ".join(f"{v:.2f}" for v in dec) + r"\\")
        tagL0 = {1.0: "A", 1.5: "B", 2.0: "C"}[L0]
        for L, v in zip(("One", "OneFive", "Two", "Three"), cov):
            macro(f"EfourCov{tagL0}{L}", f"{v:.2f}")
    lines.append(r"\bottomrule\end{tabular}")
    (TAB / "tab_drift.tex").write_text("\n".join(lines) + "\n")
    sel = [x for x, t in zip(res, tasks) if abs(t[2] - 2.0) < 1e-9]
    macro("EfourWrongPlain", f"{100 * np.mean([x['1.0']['wrong_among_declared'] for x in sel]):.1f}")
    macro("EfourWrongText", "none is" if worst == 0 else ("fewer than 0.1 percent are" if worst < 0.001 else f"{100 * worst:.1f} percent are"))
    macro("EfourReps", f"{int(len(res) / 3)}"); macro("EfourMCSE", f"{np.ceil(mc * 1000) / 1000:.3f}")


def figure_types():
    r = load("E11")
    if r is None:
        return
    res = r["results"]; concs = sorted(set(x["conc"] for x in res), reverse=True)
    sd = [np.mean([x["sd_share"] for x in res if x["conc"] == c]) for c in concs]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(wspace=0.34))
    ax = axes[0]
    for f, lab, col, mk in (("drift_median", "Median unit", MUTED, "s"), ("drift_p90", "90th percentile", BLUE, "o"), ("drift_max", "Largest", ORANGE, "^")):
        ax.plot(sd, [np.mean([x[f] for x in res if x["conc"] == c]) for c in concs], color=col, marker=mk, mec="white", mew=0.8, lw=1.3, label=lab)
    bound = res[0]["bound"]
    ax.axhline(bound, color=INK, lw=0.6, ls=(0, (2, 2))); ax.text(sd[-1], bound * 0.97, "Lemma 1 bound", ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlabel("SD of the share of motivated raters"); ax.set_ylabel(r"Realized departure, $\Lambda_j$"); ax.set_ylim(0.8, bound * 1.08)
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.1); panel(ax, "(a) How far units depart from the class norm")
    ax = axes[1]
    for f, lab, col, mk, ls in (("cov_plain", "Types pooled, plain interval", MUTED, "s", "--"), ("cov_widened_p90", "Types pooled, widened interval", AQUA, "^", "-"),
                                ("cov_stratified", "Stratified by rater type", BLUE, "o", "-")):
        ax.plot(sd, [np.mean([x[f] for x in res if x["conc"] == c]) for c in concs], color=col, marker=mk, ls=ls, mec="white", mew=0.8, lw=1.3, label=lab)
    ax.axhline(0.95, color=INK, lw=0.6); ax.set_ylim(0.8, 1.01)
    ax.set_xlabel("SD of the share of motivated raters"); ax.set_ylabel("Coverage of 95% intervals")
    ax.legend(loc="lower left", handlelength=1.8, borderaxespad=0.1); panel(ax, "(b) Coverage")
    fig.savefig(FIG / "fig_types.pdf"); plt.close(fig)
    worst = [x for x in res if x["conc"] == concs[-1]]; mid = [x for x in res if x["conc"] == 10]
    macro("ElevenReps", f"{int(len(res) / len(concs))}")
    macro("ElevenSDMax", f"{sd[-1]:.2f}"); macro("ElevenCovPlainWorst", f"{mean_of(worst, 'cov_plain'):.2f}")
    macro("ElevenCovStratWorst", f"{mean_of(worst, 'cov_stratified'):.2f}"); macro("ElevenCovWideWorst", f"{mean_of(worst, 'cov_widened_p90'):.2f}")
    macro("ElevenPninetyWorst", f"{mean_of(worst, 'drift_p90'):.1f}"); macro("ElevenPninetyMid", f"{mean_of(mid, 'drift_p90'):.1f}")
    macro("ElevenMaxWorst", f"{mean_of(worst, 'drift_max'):.1f}")
    macro("ElevenCovPlainMid", f"{mean_of(mid, 'cov_plain'):.2f}"); macro("ElevenSDMid", f"{np.mean([x['sd_share'] for x in mid]):.2f}")
    macro("ElevenRhoPooledWorst", f"{mean_of(worst, 'rho_pooled'):.2f}"); macro("ElevenRhoStratWorst", f"{mean_of(worst, 'rho_stratified'):.2f}")


def figure_class_size():
    r = load("E12")
    if r is None:
        return
    res = [x for x in r["results"] if not x["failed"]]; Js = sorted(set(x["J"] for x in res))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), gridspec_kw=dict(wspace=0.36))
    cov = [np.mean([x["coverage"] for x in res if x["J"] == J]) for J in Js]
    tau = [np.mean([x["tau"] for x in res if x["J"] == J]) for J in Js]
    axes[0].plot(Js, cov, color=BLUE, marker="o", mec="white", mew=0.8); axes[0].axhline(0.95, color=INK, lw=0.6)
    axes[0].set_xscale("log"); axes[0].set_xticks(Js); axes[0].set_xticklabels([str(J) for J in Js]); axes[0].minorticks_off(); axes[0].set_ylim(0.85, 1.0)
    axes[0].set_xlabel("Units in the comparison class"); axes[0].set_ylabel("Coverage of 95% intervals"); panel(axes[0], "(a) Calibration")
    axes[1].plot(Js, tau, color=BLUE, marker="o", mec="white", mew=0.8); axes[1].axhline(0.45, color=INK, lw=0.6)
    axes[1].set_xscale("log"); axes[1].set_xticks(Js); axes[1].set_xticklabels([str(J) for J in Js]); axes[1].minorticks_off(); axes[1].set_ylim(0.35, 0.55)
    axes[1].set_xlabel("Units in the comparison class"); axes[1].set_ylabel(r"Mean of $\hat\tau$ (truth 0.45)"); panel(axes[1], "(b) Spread of the index")
    fig.savefig(FIG / "fig_class_size.pdf"); plt.close(fig)
    for J, c in zip(Js, cov):
        macro("TwelveCov" + {15: "Fifteen", 30: "Thirty", 60: "Sixty", 125: "OneTwentyFive", 250: "TwoFifty", 500: "FiveHundred", 1000: "Thousand"}[J], f"{c:.2f}")
    macro("TwelveReps", f"{int(len(r['results']) / len(Js))}")


def tables_misspec_volume_nested():
    r5, r6, r8 = load("E5"), load("E6"), load("E8")
    if r5 is not None:
        tasks = np.array(r5["tasks"]); res = r5["results"]
        lines = [r"\begin{tabular}{@{}ll ccc ccc@{}}\toprule",
                 r" & & \multicolumn{3}{c}{Rank correlation with latent mean} & \multicolumn{3}{c}{Population-level stability}\\\cmidrule(lr){3-5}\cmidrule(l){6-8}",
                 r"Latent law & Visibility & Mean stars & Bayes avg. & Index & Slope, mean & Slope, index & Rank agreement\\\midrule"]
        m5 = 0.0
        for h, lab in ((0.0, "Probit, common scale"), (1.0, "Probit, unit-specific scale")):
            sel = [x for x, t in zip(res, tasks) if abs(t[2] - h) < 1e-9]
            for i, g in enumerate(("none", "negativity", "J-shaped")):
                row = [np.mean([x[g][k] for x in sel]) for k in ("raw", "bayes", "index")]
                for k in ("raw", "bayes", "index"):
                    m5 = max(m5, np.std([x[g][k] for x in sel], ddof=1) / np.sqrt(len(sel)))
                extra = ([np.mean([x["pop_slope_mean"] for x in sel]), np.mean([x["pop_slope_index"] for x in sel]),
                          np.mean([x["pop_rank_agree_index"] for x in sel])] if g == "J-shaped" else None)
                lines.append((lab if i == 0 else "") + f" & {g} & " + " & ".join(f"{v:.2f}" for v in row) + " & " +
                             (" & ".join(f"{v:.2f}" for v in extra) if extra else " & & ") + r"\\")
            tag = "Hom" if h == 0 else "Het"
            macro(f"EfiveSlopeMean{tag}", f"{np.mean([x['pop_slope_mean'] for x in sel]):.2f}")
            macro(f"EfiveSlopeIndex{tag}", f"{np.mean([x['pop_slope_index'] for x in sel]):.2f}")
            macro(f"EfiveRankAgree{tag}", f"{np.mean([x['pop_rank_agree_index'] for x in sel]):.2f}")
        lines.append(r"\bottomrule\end{tabular}")
        (TAB / "tab_misspecification.tex").write_text("\n".join(lines) + "\n")
        macro("EfiveMCSE", f"{np.ceil(m5 * 1000) / 1000:.3f}"); macro("EfiveReps", f"{int(len(res) / 2)}")
    if r6 is not None:
        res = r6["results"]
        lines = [r"\begin{tabular}{@{}lccccccc@{}}\toprule",
                 r"Rating counts & $\hat\gamma$ & SD & Mean SE & Coverage & Realized contrast & $\hat\tau$ & Corr$(\theta_j,\log N_j)$\\\midrule"]
        for k, lab in (("exogenous", r"Unrelated to $\theta_j$"), ("endogenous", "As implied by J-shaped visibility")):
            v = {m: np.mean([x[k][m] for x in res]) for m in ("gamma", "gamma_se", "cover_gamma", "truth", "tau", "corr_theta_logN")}
            sd = np.std([x[k]["gamma"] for x in res], ddof=1)
            lines.append(f"{lab} & {v['gamma']:.3f} & {sd:.3f} & {v['gamma_se']:.3f} & {v['cover_gamma']:.2f} & {v['truth']:.3f} & {v['tau']:.3f} & {v['corr_theta_logN']:+.2f}" + r"\\")
            macro(f"Esix{k.capitalize()}Gamma", f"{v['gamma']:.3f}"); macro(f"Esix{k.capitalize()}Tau", f"{v['tau']:.3f}")
            macro(f"Esix{k.capitalize()}Cover", f"{v['cover_gamma']:.2f}")
        lines.append(r"\quad with a quadratic in $\log N_j$ in the prior mean & " + f"{np.mean([x['endogenous']['gamma_volume_terms'] for x in res]):.3f}" + r" & & & & & & \\")
        gi = np.mean([x["endogenous"]["gamma_interacted"] for x in res])
        lines.append(r"\quad and its interaction with the covariate (average contrast) & " + f"{gi:.3f}" + r" & & & & & & \\")
        lines.append(r"\bottomrule\end{tabular}")
        (TAB / "tab_volume.tex").write_text("\n".join(lines) + "\n")
        macro("EsixInteracted", f"{gi:.3f}")
        macro("EsixEndogenousExcess", f"{100 * (np.mean([x['endogenous']['gamma'] for x in res]) / 0.30 - 1):.0f}")
        macro("EsixEndogenousExcessTau", f"{100 * (np.mean([x['endogenous']['tau'] for x in res]) / 0.45 - 1):.0f}")
        macro("EsixBiasInSE", f"{(np.mean([x['endogenous']['gamma'] for x in res]) - 0.30) / np.mean([x['endogenous']['gamma_se'] for x in res]):.1f}")
        m6 = max(np.std([x[k][m] for x in res], ddof=1) / np.sqrt(len(res)) for k in ("exogenous", "endogenous") for m in ("gamma", "tau"))
        macro("EsixMCSE", f"{np.ceil(m6 * 1000) / 1000:.3f}"); macro("EsixReps", f"{len(res)}")
    if r8 is not None:
        res = r8["results"]
        for k, tag in (("rho_group", "Group"), ("rho_pooled", "Pooled"), ("rho_unitavg", "UnitAvg"), ("rho_bayesavg", "BayesAvg"),
                       ("cover_group", "CoverGroup"), ("cover_unit", "CoverUnit"), ("tau", "Tau"), ("omega", "Omega")):
            macro(f"Eeight{tag}", f"{np.mean([x[k] for x in res]):.2f}")
        macro("EeightReps", f"{len(res)}")


def realized_medians():
    """Median number of visible ratings among rated units, one replication per design (for Table C1)."""
    import run_simulations as R
    rng = np.random.default_rng([R.SEED, 99])
    def med(d):
        n = d["X"].sum(1); return int(round(np.median(n[n > 0])))
    macro("MedSix", f"{med(simulate_class(2000, rng, c=C_LATENT, b=B_J, tau=0.45))}")
    macro("MedEight", f"{med(simulate_class(2000, rng, c=C_LATENT, b=B_J, tau=0.45, med_n=8.0))}")
    macro("MedTwenty", f"{med(simulate_class(2000, rng, c=C_LATENT, b=B_J, tau=0.45, med_n=20.0))}")
    macro("MedTwentyMod", f"{med(simulate_class(2000, rng, c=C_LATENT, b=B_J, tau=0.45, med_n=20.0, retain=np.array([0.5, 0.7, 1, 1, 1.0])))}")
    r = load("E2")
    if r is not None:
        macro("MedEtwo", f"{np.median([x['median_n'] for x in r['results']]):.0f}")


if __name__ == "__main__":
    figure_identification()
    realized_medians()
    figure_ladder()
    theory_numbers()
    figure_scorecard()
    figure_manipulation()
    figure_sparse_robustness()
    figure_invariance()
    table_sources()
    table_drift()
    figure_types()
    figure_class_size()
    tables_misspec_volume_nested()
    (TAB / "macros.tex").write_text("% Generated by code/make_exhibits.py; do not edit.\n" +
                                   "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n")
    print(f"wrote {len(MACROS)} macros")
