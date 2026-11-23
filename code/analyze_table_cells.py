# /*
#  * L-SHADE-SSR: A Hybrid Success-History Differential Evolution with Staggered Sub-population Restarts
#  * Authors: Chau Van Van (1,2,*), Nguyen Minh Tuan (1), Chau Khai Kien (3), and Phayung Meesad (2,*)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
#  *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
#  *     King Mongkut's University of Technology North Bangkok,
#  *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok 10800, Thailand
#  *     phayung.m@itd.kmutnb.ac.th
#  * (3) Faculty of Artificial Intelligence, Asia University Viet Nam,
#  *     No. 485 Le Quang Dinh, Ho Chi Minh City, Vietnam
#  *     khaikien0763@gmail.com
#  * (*) Corresponding authors
#  * Repository: https://github.com/vancv43/LSHADE-SSR-Mechanical-Design (MIT License)
#  */
"""Table cells of the paper that the other analysis scripts do not write out:
(a) Table of the primary test: median A12 over the 82 and the 70 cases (CEC2022 D10 block removed) and the
    unadjusted sign-test p of the 70 cases;
(b) BBOB table: the untuned L-SHADE-SSR against the tuned L-SHADE-T and jSO-T, rank-sum W/T/L with Holm over the
    24 functions and the signed-rank p over the 24 mean errors, at D = 20 and 40; and the descriptive median
    A12 of L-SHADE-SSR-T against L-SHADE-T and jSO-T over the functions not solved to the 1e-8 floor by every run.
Output: results/analysis/table_cells_primary.csv, table_cells_bbob.csv, table_cells_bbob_a12.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analyze_rev1_b as B
import analyze_rev2 as R2
from analyze_rev1_c import wtl, paired, a12
OUT = os.path.join(HERE, 'results', 'analysis')


def primary():
    d = B.load_all()
    pre = d[~((d.suite == 'cec2017') & (d.dim == 50))]
    rows = []
    for label, g in [('82 cases', pre), ('70 cases', pre[~((pre.suite == 'cec2022') & (pre.dim == 10))])]:
        C = g.groupby(['suite', 'dim', 'function', 'algorithm']).error.mean().unstack()
        for o in ['MPA', 'CS', 'ABC']:
            A = [B.a12(g[(g.suite == s) & (g.dim == D) & (g.function == f) & (g.algorithm == 'L-SHADE-SSR')].error,
                       g[(g.suite == s) & (g.dim == D) & (g.function == f) & (g.algorithm == o)].error) for s, D, f in C.index]
            bt, wo = int((C['L-SHADE-SSR'] < C[o]).sum()), int((C['L-SHADE-SSR'] > C[o]).sum())
            rows.append(dict(cases=label, n=len(C), opponent=o, better=bt, worse=wo, tie=len(C) - bt - wo,
                             sign_p_unadjusted=stats.binomtest(bt, bt + wo).pvalue, a12_median=float(np.median(A))))
            print(f'{label} vs {o}: B/W/T {bt}/{wo}/{len(C) - bt - wo}  sign p={rows[-1]["sign_p_unadjusted"]:.3g}  '
                  f'median A12={rows[-1]["a12_median"]:.2f}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'table_cells_primary.csv'), index=False)


def bbob():
    d = R2.load(); M = d.groupby(['dim', 'function', 'algorithm']).error.mean().unstack(); rows = []
    for D in (20, 40):
        g = d[d.dim == D]
        for b in ('L-SHADE-T', 'jSO-T'):
            w, t, l, won, lost = wtl(g, 'L-SHADE-SSR', b, R2.FNS); bt, wo, ti, sr, sg = paired(M.loc[D], 'L-SHADE-SSR', b)
            rows.append(dict(dim=D, a='L-SHADE-SSR', b=b, win=w, tie=t, loss=l, better=bt, worse=wo, tie_mean=ti, signed_rank_p=sr, sign_p=sg))
            print(f'D{D} L-SHADE-SSR vs {b}: W/T/L {w}/{t}/{l}  signed-rank p={sr:.3g}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'table_cells_bbob.csv'), index=False)
    rows = []
    for D in (20, 40):
        g = d[d.dim == D]
        fs = [f for f in R2.FNS if not (g[g.function == f].error == 0).all()]
        for b in ('L-SHADE-T', 'jSO-T'):
            A = [a12(g[(g.function == f) & (g.algorithm == 'L-SHADE-SSR-T')].error, g[(g.function == f) & (g.algorithm == b)].error) for f in fs]
            rows.append(dict(dim=D, a='L-SHADE-SSR-T', b=b, functions_not_solved=len(fs), a12_median=float(np.median(A))))
            print(f'D{D} L-SHADE-SSR-T vs {b}: median A12 over {len(fs)} unsolved functions {rows[-1]["a12_median"]:.2f}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'table_cells_bbob_a12.csv'), index=False)


if __name__ == '__main__':
    primary()
    bbob()
