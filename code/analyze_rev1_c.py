# /*
#  * L-SHADE-SSR: A Hybrid Success-History Differential Evolution with Staggered Sub-population Restarts
#  * Authors: Chau Van Van (1,3,*), Nguyen Minh Tuan (1), Chau Khai Kien (2), and Phayung Meesad (3,*)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
#  *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Faculty of Artificial Intelligence, Asia University Viet Nam,
#  *     No. 485 Le Quang Dinh, Ho Chi Minh City, Vietnam
#  *     khaikien0763@gmail.com
#  * (3) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
#  *     King Mongkut's University of Technology North Bangkok,
#  *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok, 10800, Thailand
#  *     phayung.m@itd.kmutnb.ac.th
#  * (*) Corresponding authors
#  * Repository: https://github.com/vancv43/LSHADE-SSR-Mechanical-Design (MIT License)
#  */
"""Analysis of the reviewer-round runs (grid rev1), exactly as fixed in PROTOCOL_power.md, Deviations 2026-10-22.
(a) L-SHADE-SSR against L-SHADE and jSO on CEC2017 at D = 10, 30, 50: signed-rank on mean errors plus sign test, Holm over the
    two opponents, for the 58 D10+D30 cases and for each dimension; per-function rank-sum W/T/L with Holm over
    functions; Vargha-Delaney A12; Friedman with Holm post-hoc, L-SHADE-SSR as control.
(b) each D = 30 ablation variant against L-SHADE-SSR: rank-sum W/T/L with Holm over the 29 functions, signed-rank over the
    29 mean errors with Holm over the eight variants.
(c) L-SHADE-SSR-N0fixed against MPA, CS and ABC at D = 30 and 50 with the D = 50 procedure, and against L-SHADE-SSR.
Reported whatever it shows. Output: results/analysis/rev1_c_*.csv and rev1_c_summary.txt"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "analysis"); L = []
ABL30 = ['v4-noRestart', 'v4-noGate', 'v4-noMacro', 'v4-sharedMem', 'v4-K1', 'v4-equalPeriods', 'v4-gaussSteps', 'v4-restart']


def say(s=""):
    print(s); L.append(str(s))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def a12(x, y):
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def read(pattern, cols=("suite", "dim", "algorithm", "function", "run", "error")):
    fs = glob.glob(os.path.join(ROOT, pattern))
    return pd.concat([pd.read_csv(f)[list(cols)] for f in fs]) if fs else pd.DataFrame(columns=cols)


def load():
    new = read("runs/rev1/out/*/results/rev1-test_s*.csv")
    old = pd.concat([read("runs/power/out/*/results/power-test_s*.csv"), read("runs/power50/out/*/results/power50-test_s*.csv")])
    old = old[old.suite == "cec2017"]
    d = pd.concat([new, old], ignore_index=True)
    d = d.drop_duplicates(["dim", "algorithm", "function", "run"])
    d.loc[d.error < 1e-8, "error"] = 0.0
    return d, new


def completeness(new):
    cnt = new.groupby(["dim", "algorithm"]).size()
    say("rev1 runs on disk (870 = complete per dimension and algorithm):")
    for (dim, a), n in cnt.items():
        say(f"  D{dim} {a:16s} {n}")
    return cnt


def wtl(g, a, b, fns):
    """W/T/L of a against b over functions, rank-sum with Holm over functions (W = a significantly lower error)"""
    ps, dirs = [], []
    for f in fns:
        x = g[(g.algorithm == a) & (g.function == f)].error.values; y = g[(g.algorithm == b) & (g.function == f)].error.values
        ps.append(1.0 if np.array_equal(np.sort(x), np.sort(y)) else stats.ranksums(x, y).pvalue)
        dirs.append(np.sign(np.median(y) - np.median(x)) or np.sign(y.mean() - x.mean()))
    q = holm(ps); dirs = np.array(dirs)
    w = int(((q < .05) & (dirs > 0)).sum()); l = int(((q < .05) & (dirs < 0)).sum())
    lost = " ".join(f for f, qq, s in zip(fns, q, dirs) if qq < .05 and s < 0)
    won = " ".join(f for f, qq, s in zip(fns, q, dirs) if qq < .05 and s > 0)
    return w, len(fns) - w - l, l, won, lost


def paired(m, a, b):
    """signed-rank and sign test on mean errors of a vs b over the rows of m; B/W/T counts from a's side"""
    bt = int((m[a] < m[b]).sum()); wo = int((m[a] > m[b]).sum()); ti = int((m[a] == m[b]).sum())
    sr = stats.wilcoxon(m[a], m[b]).pvalue if (m[a] != m[b]).any() else 1.0
    sg = stats.binomtest(bt, bt + wo).pvalue if bt + wo else 1.0
    return bt, wo, ti, sr, sg


def friedman(g, algs, fns):
    M = np.array([[g[(g.algorithm == a) & (g.function == f)].error.mean() for a in algs] for f in fns])
    R = np.vstack([stats.rankdata(r) for r in M]); avg = R.mean(0)
    chi2, p = stats.friedmanchisquare(*M.T); n, k = M.shape
    se = np.sqrt(k * (k + 1) / (6 * n)); pz = 2 * stats.norm.sf(np.abs((avg - avg[0]) / se))
    return dict(zip(algs, avg)), dict(zip(algs[1:], holm(pz[1:]))), chi2, p


def part_a(d):
    say("\n=== (a) L-SHADE-SSR vs L-SHADE and jSO on CEC2017")
    opp = ["L-SHADE", "jSO"]; rows = []; means = {}
    for dim in (10, 30, 50):
        g = d[(d.dim == dim) & d.algorithm.isin(["L-SHADE-SSR"] + opp)]
        fns = sorted(g.function.unique(), key=lambda s: int(s[1:]))
        if g.groupby("algorithm").size().reindex(["L-SHADE-SSR"] + opp).fillna(0).min() < 870:
            say(f"  D{dim}: incomplete, skipped"); continue
        m = g.groupby(["function", "algorithm"]).error.mean().unstack(); means[dim] = m
        avg, hp, chi2, p = friedman(g, ["L-SHADE-SSR"] + opp, fns)
        say(f"  D{dim}: Friedman chi2={chi2:.2f} p={p:.3g}; avg rank " + ", ".join(f"{a} {avg[a]:.2f}" for a in ["L-SHADE-SSR"] + opp))
        for o in opp:
            w, t, l, won, lost = wtl(g, "L-SHADE-SSR", o, fns); bt, wo, ti, sr, sg = paired(m, "L-SHADE-SSR", o)
            A = np.median([a12(g[(g.function == f) & (g.algorithm == "L-SHADE-SSR")].error, g[(g.function == f) & (g.algorithm == o)].error) for f in fns])
            rows.append(dict(dim=dim, opponent=o, avg_rank_ssr=avg["L-SHADE-SSR"], avg_rank_opp=avg[o], friedman_holm_p=hp[o], win=w, tie=t, loss=l,
                             better=bt, worse=wo, tie_mean=ti, signed_rank_p=sr, sign_p=sg, a12_median=A, won_on=won, lost_on=lost))
            say(f"    vs {o:7s}: post-hoc Holm p={hp[o]:.3g}  W/T/L {w}/{t}/{l}  B/W/T {bt}/{wo}/{ti}  signed-rank p={sr:.3g}  sign p={sg:.3g}  "
                f"A12={A:.2f}  lost on {lost or '-'}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev1_c_shade_blocks.csv"), index=False)
    if 10 in means and 30 in means:
        C = pd.concat([means[10], means[30]]); res = []
        for o in opp:
            res.append(paired(C, "L-SHADE-SSR", o))
        adj = holm([r[3] for r in res])
        say(f"  pooled 58 cases (D10+D30), Holm over the two opponents:")
        for o, r, q in zip(opp, res, adj):
            say(f"    vs {o:7s}: B/W/T {r[0]}/{r[1]}/{r[2]}  signed-rank Holm p={q:.3g}  sign p={r[4]:.3g}")
    if all(k in means for k in (10, 30, 50)):
        for dim in (10, 30, 50):
            res = [paired(means[dim], "L-SHADE-SSR", o) for o in opp]; adj = holm([r[3] for r in res])
            say(f"  D{dim} signed-rank Holm over two: " + ", ".join(f"{o} p={q:.3g}" for o, q in zip(opp, adj)))


def part_b(d):
    say("\n=== (b) restart ablation and replacement controls, CEC2017 D30, each variant against L-SHADE-SSR")
    g = d[(d.dim == 30) & d.algorithm.isin(["L-SHADE-SSR"] + ABL30)]
    fns = sorted(g.function.unique(), key=lambda s: int(s[1:]))
    have = [a for a in ABL30 if (g.algorithm == a).sum() == 870]
    if not have:
        say("  no complete variant yet"); return
    m = g.groupby(["function", "algorithm"]).error.mean().unstack(); rows = []
    for a in have:
        w, t, l, won, lost = wtl(g, "L-SHADE-SSR", a, fns); bt, wo, ti, sr, sg = paired(m, "L-SHADE-SSR", a)
        rows.append(dict(variant=a, ssr_win=w, tie=t, ssr_loss=l, ssr_better=bt, ssr_worse=wo, tie_mean=ti, signed_rank_p=sr, sign_p=sg,
                         ssr_won_on=won, ssr_lost_on=lost))
    adj = holm([r["signed_rank_p"] for r in rows])
    for r, q in zip(rows, adj):
        r["signed_rank_holm_p"] = q
        say(f"  {r['variant']:16s} L-SHADE-SSR W/T/L {r['ssr_win']}/{r['tie']}/{r['ssr_loss']}  B/W/T {r['ssr_better']}/{r['ssr_worse']}/{r['tie_mean']}  "
            f"signed-rank Holm p={q:.3g}  L-SHADE-SSR better on {r['ssr_won_on'] or '-'}; worse on {r['ssr_lost_on'] or '-'}")
    if len(have) < len(ABL30):
        say(f"  incomplete: {sorted(set(ABL30) - set(have))}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev1_c_ablation30.csv"), index=False)
    g.groupby(["function", "algorithm"]).error.agg(["mean", "std"]).to_csv(os.path.join(OUT, "rev1_c_ablation30_mean.csv"))


def part_c(d):
    say("\n=== (c) L-SHADE-SSR with a fixed initial population of 110")
    rows = []
    for dim in (30, 50):
        g = d[(d.dim == dim) & d.algorithm.isin(["L-SHADE-SSR-N0fixed", "L-SHADE-SSR", "MPA", "CS", "ABC"])]
        if (g.algorithm == "L-SHADE-SSR-N0fixed").sum() < 870:
            say(f"  D{dim}: incomplete, skipped"); continue
        fns = sorted(g.function.unique(), key=lambda s: int(s[1:]))
        m = g.groupby(["function", "algorithm"]).error.mean().unstack()
        res = [paired(m, "L-SHADE-SSR-N0fixed", o) for o in ["MPA", "CS", "ABC"]]; adj = holm([r[3] for r in res])
        for o, r, q in zip(["MPA", "CS", "ABC"], res, adj):
            w, t, l, won, lost = wtl(g, "L-SHADE-SSR-N0fixed", o, fns)
            rows.append(dict(dim=dim, opponent=o, win=w, tie=t, loss=l, better=r[0], worse=r[1], tie_mean=r[2], signed_rank_holm_p=q, sign_p=r[4]))
            say(f"  D{dim} N0fixed vs {o}: W/T/L {w}/{t}/{l}  B/W/T {r[0]}/{r[1]}/{r[2]}  signed-rank Holm p={q:.3g}  sign p={r[4]:.3g}")
        w, t, l, won, lost = wtl(g, "L-SHADE-SSR", "L-SHADE-SSR-N0fixed", fns); bt, wo, ti, sr, sg = paired(m, "L-SHADE-SSR", "L-SHADE-SSR-N0fixed")
        rows.append(dict(dim=dim, opponent="L-SHADE-SSR (11D)", win=w, tie=t, loss=l, better=bt, worse=wo, tie_mean=ti, signed_rank_holm_p=sr, sign_p=sg))
        say(f"  D{dim} L-SHADE-SSR (N0=11D) vs N0fixed: W/T/L {w}/{t}/{l}  B/W/T {bt}/{wo}/{ti}  signed-rank p={sr:.3g}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev1_c_n0fixed.csv"), index=False)


if __name__ == "__main__":
    d, new = load()
    completeness(new)
    part_a(d); part_b(d); part_c(d)
    open(os.path.join(OUT, "rev1_c_summary.txt"), "w").write("\n".join(L) + "\n")
