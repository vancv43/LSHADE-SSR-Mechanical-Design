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
"""
Component-switchable L-SHADE-SSR for the ablation study.

With all switches at their defaults, `ssr_switchable` reproduces
algorithms.lshade_ssr_v2(..., target_restarts=20) exactly (same RNG draw order);
this is checked by `_self_test()` below. The final L-SHADE-SSR evaluated in the
manuscript is that configuration, frozen before the CEC2017 runs.

Switches (each isolates one component):
  use_restart  False -> subpops never restart (inner-search search only);
                  PRR can then never fire either.
  use_crossover     False -> no intra-subpop arithmetic crossover after a jump.
  use_prr        False -> stagnation never changes a subpop's period/phase
                  and never reseeds the best.
  reseed_best   False -> PRR changes period/phase only (best kept).
  prime_periods  False -> initial periods are plain integers (not rounded to
                  primes) and PRR moves to the next unused integer with no
                  coprimality test. Tests whether primality matters at all.
  n_subpops       1 -> a single subpop (no independently clocked subpops).
  trace          optional callback(t, X, fit, subpop_id, restarted, prr_fired)
                  called after each iteration; observes only.
  sync_clocks    True -> all subpops share one period and one phase, so they
                  restart together (control for staggering; used with
                  use_prr=False in run_sync_control.py, not in ABLATIONS).
"""
import numpy as np
import sympy

from algorithms import _bounds, _clip, _init_pop, _levy, _next_prime_coprime

_SPREAD = [0.55, 0.75, 1.0, 1.35, 1.7, 2.1, 2.6, 3.2]


def _initial_periods(max_iter, n_subpops, target_restarts, prime_periods):
    base = max(int(max_iter / target_restarts), 5)
    periods, used = [], set()
    for s in _SPREAD[:n_subpops]:
        if prime_periods:
            cand = sympy.nextprime(max(int(base * s), 5) - 1)
            while cand in used:
                cand = sympy.nextprime(cand)
        else:
            cand = max(int(base * s), 5)
            while cand in used:
                cand += 1
        used.add(cand)
        periods.append(cand)
    return np.array(periods, dtype=int)


def _next_plain(p_current, used):
    cand = p_current + 1
    while cand in used:
        cand += 1
    return cand


def ssr_switchable(func, lb, ub, dim, pop_size, max_iter, seed=0,
                   n_subpops=4, stagnation_threshold=3, eta0=2.2, L0=0.25,
                   gamma=4.0, target_restarts=20, rel_eps=1e-4,
                   use_restart=True, use_crossover=True, use_prr=True,
                   reseed_best=True, prime_periods=True, sync_clocks=False,
                   trace=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)

    subpop_id = np.array([i % n_subpops for i in range(pop_size)])
    periods = _initial_periods(max_iter, n_subpops, target_restarts, prime_periods)
    phases = rng.integers(0, periods, size=n_subpops)
    if sync_clocks:
        # control for staggering: every subpop shares the median subpop's
        # period and subpop 0's phase; drawn after the default RNG calls so
        # sync_clocks=False keeps the frozen RNG order bit for bit
        common = periods[min(2, n_subpops - 1)]
        periods = np.full(n_subpops, common, dtype=int)
        phases = np.full(n_subpops, phases[0] % common, dtype=int)
    subpop_best_fit = np.array([fit[subpop_id == b].min() for b in range(n_subpops)])
    stagnation = np.zeros(n_subpops, dtype=int)
    last_improve_iter = np.zeros(n_subpops, dtype=int)
    used_periods = set(periods.tolist())

    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    prr_triggers = 0

    for t in range(max_iter):
        restarted, prr_fired = [], []
        for b in range(n_subpops):
            members = np.where(subpop_id == b)[0]
            restarting = use_restart and ((t - phases[b]) % periods[b] == 0)
            if restarting:
                restarted.append(b)
            if not restarting:
                depth_frac = ((t - phases[b]) % periods[b]) / periods[b]
                eta = eta0 * (0.25 + 0.75 * (1.0 - depth_frac)) * (1.0 - 0.7 * t / max_iter)
                b_best = X[members[np.argmin(fit[members])]]
                r1 = rng.uniform(0, 1, size=(len(members), dim))
                r2 = rng.uniform(0, 1, size=(len(members), dim))
                X[members] = X[members] + eta * r1 * (b_best - X[members]) \
                                          + eta * r2 * (gbest - X[members])
                X[members] = _clip(X[members], lb, ub)
                fit[members] = func(X[members])
            else:
                best = members[np.argmin(fit[members])]
                restarting_members = members[members != best]
                amp = L0 * np.exp(-gamma * t / max_iter)
                for i in restarting_members:
                    levy = np.clip(_levy(dim, rng), -3, 3)
                    X[i] = X[best] + amp * (ub - lb) * levy
                X[restarting_members] = _clip(X[restarting_members], lb, ub)
                fit[restarting_members] = func(X[restarting_members])
                if use_crossover:
                    shuffled = rng.permutation(restarting_members)
                    for k in range(0, len(shuffled) - 1, 2):
                        i, j = shuffled[k], shuffled[k + 1]
                        lam = rng.uniform()
                        child = _clip(lam * X[i] + (1 - lam) * X[j], lb, ub)
                        child_fit = func(child.reshape(1, -1))[0]
                        worse = i if fit[i] > fit[j] else j
                        if child_fit < fit[worse]:
                            X[worse] = child
                            fit[worse] = child_fit
                new_best = fit[members].min()
                rel_improve = (subpop_best_fit[b] - new_best) / (abs(subpop_best_fit[b]) + 1e-12)
                if rel_improve > rel_eps:
                    subpop_best_fit[b] = new_best
                    stagnation[b] = 0
                    last_improve_iter[b] = t
                else:
                    stagnation[b] += 1
                if use_prr and stagnation[b] >= stagnation_threshold:
                    prr_triggers += 1
                    prr_fired.append(b)
                    tau = max(t - last_improve_iter[b], periods[b])
                    used_periods.discard(periods[b])
                    if prime_periods:
                        new_p = _next_prime_coprime(periods[b], tau, used_periods)
                    else:
                        new_p = _next_plain(periods[b], used_periods)
                    used_periods.add(new_p)
                    periods[b] = new_p
                    phases[b] = rng.integers(0, periods[b])
                    stagnation[b] = 0
                    if reseed_best:
                        best_new = rng.uniform(lb, ub)
                        X[best] = best_new
                        fit[best] = func(best_new.reshape(1, -1))[0]
                        subpop_best_fit[b] = fit[members].min()
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
        if trace is not None:
            # read-only observer for behavior figures; draws no random numbers
            trace(t, X, fit, subpop_id, restarted, prr_fired)
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve,
                prr_triggers=prr_triggers)


def ssr_final(func, lb, ub, dim, pop_size, max_iter, seed=0):
    """The frozen L-SHADE-SSR configuration reported in the manuscript."""
    return ssr_switchable(func, lb, ub, dim, pop_size, max_iter, seed=seed)


ABLATIONS = {
    "L-SHADE-SSR (full)": {},
    "w/o restart": dict(use_restart=False),
    "w/o crossover": dict(use_crossover=False),
    "w/o PRR": dict(use_prr=False),
    "w/o best reseed": dict(reseed_best=False),
    "non-prime periods": dict(prime_periods=False),
    "single subpop": dict(n_subpops=1),
    "sparse schedule (E=7)": dict(target_restarts=7),
}


def _self_test():
    from algorithms import lshade_ssr_v2
    from benchmarks import BENCHMARKS
    for fn in ["F1", "F9", "F21"]:
        func, lb, ub, dim, _ = BENCHMARKS[fn]
        for seed in range(3):
            a = lshade_ssr_v2(func, lb, ub, dim, pop_size=30, max_iter=500, seed=seed,
                       target_restarts=20)["best_fitness"]
            b = ssr_final(func, lb, ub, dim, pop_size=30, max_iter=500, seed=seed)["best_fitness"]
            assert a == b, (fn, seed, a, b)
    print("self-test OK: ssr_final == lshade_ssr_v2(target_restarts=20)")


if __name__ == "__main__":
    _self_test()
