from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import qr
from scipy.optimize import line_search

N, M, SPARSITY = 4096, 1024, 160
SIGMA, TRIALS, SUPPORT_EPS = 1e-2, 10, 1e-3
MU_GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
OUT_DIR = Path('../output/prob3')


def make_data(seed, n=N, m=M, sparsity=SPARSITY):
    rng = np.random.default_rng(seed)
    # QR(A.T) is the shortest stable way to give a Gaussian matrix orthonormal rows.
    a = qr(rng.standard_normal((n, m)), mode='economic', overwrite_a=True)[0].T
    x = np.zeros(n)
    support = rng.choice(n, sparsity, replace=False)
    x[support] = rng.choice((-1.0, 1.0), sparsity)
    b = a @ x + rng.normal(0.0, SIGMA, m)
    return a, b, x


def objective(a, b, tau, x):
    return 0.5 * np.linalg.norm(a @ x - b) ** 2 + tau * np.abs(x).sum()


def prp(a, b, tau, maxiter=500):
    x = np.zeros(a.shape[1])
    history, iterations, restarts = [], 0, 0
    ata_diag = np.sum(a * a, axis=0)
    for mu in MU_GRID:
        def fun(z):
            return 0.5 * np.linalg.norm(a @ z - b) ** 2 + tau * np.sqrt(z * z + mu * mu).sum()

        def grad(z):
            return a.T @ (a @ z - b) + tau * z / np.sqrt(z * z + mu * mu)

        def precondition(g, z):
            return g / (ata_diag + tau * mu * mu / (z * z + mu * mu) ** 1.5)

        g, f = grad(x), fun(x)
        p = precondition(g, x)
        d = -p
        tol = max(1e-7, mu * 1e-2)
        for _ in range(maxiter):
            if np.linalg.norm(g, np.inf) <= tol:
                break
            if g @ d > -1e-4 * (g @ p):
                d, restarts = -p, restarts + 1
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                step = line_search(fun, grad, x, d, gfk=g, old_fval=f,
                                   c1=1e-4, c2=0.4, maxiter=20)[0]
            if step is None:
                step = 1.0
                while fun(x + step * d) > f + 1e-4 * step * (g @ d):
                    step *= 0.5
            xn, gn = x + step * d, grad(x + step * d)
            pn = precondition(gn, xn)
            beta = max(0.0, float(pn @ (gn - g) / max(g @ p, 1e-30)))
            x, g, p, d, f = xn, gn, pn, -pn + beta * d, fun(xn)
            iterations += 1
            history.append(objective(a, b, tau, x))
        if np.linalg.norm(g, np.inf) > tol:
            raise RuntimeError(f'PRP+ did not converge at mu={mu:g}')
    return x, iterations, restarts, history


def gpsr(a, b, tau, maxiter=1500):
    """原文单调 GPSR-BB：投影 BB 方向、区间精确线搜索和 LCP 停止准则。"""
    n, alpha = a.shape[1], 1.0
    z = np.zeros(2 * n)
    atb = a.T @ b
    g = np.r_[-atb + tau, atb + tau]
    history = []
    for k in range(1, maxiter + 1):
        delta = np.maximum(z - alpha * g, 0.0) - z
        ad = a @ (delta[:n] - delta[n:])
        gamma = ad @ ad
        lam = 1.0 if gamma == 0 else float(np.clip(-(delta @ g) / gamma, 0.0, 1.0))
        zn = z + lam * delta
        xn = zn[:n] - zn[n:]
        atr = a.T @ (a @ xn - b)
        gn = np.r_[atr + tau, -atr + tau]
        history.append(objective(a, b, tau, xn))
        if np.linalg.norm(np.minimum(zn, gn)) <= 1e-2:
            return xn, k, 0, history
        alpha = 1e30 if gamma == 0 else float(np.clip((delta @ delta) / gamma, 1e-30, 1e30))
        z, g = zn, gn
    return z[:n] - z[n:], maxiter, 0, history


def metrics(xhat, x):
    error = np.linalg.norm(xhat - x)
    truth, found = np.abs(x) > 0, np.abs(xhat) > SUPPORT_EPS
    tp = np.count_nonzero(truth & found)
    precision = tp / max(1, found.sum())
    recall = tp / truth.sum()
    return error / np.linalg.norm(x), error * error / x.size, \
        20 * np.log10(np.linalg.norm(x) / error), \
        2 * precision * recall / max(precision + recall, 1e-30)


def run(seed):
    a, b, x = make_data(seed)
    tau = 0.1 * np.linalg.norm(a.T @ b, np.inf)
    rows, curves = [], {}
    for name, solver in (('PRP+', prp), ('GPSR-BB', gpsr)):
        start = perf_counter()
        xhat, iterations, restarts, history = solver(a, b, tau)
        elapsed = perf_counter() - start
        rel, mse, snr, f1 = metrics(xhat, x)
        rows.append((seed, name, iterations, restarts, elapsed,
                     objective(a, b, tau, xhat), rel, mse, snr, f1))
        curves[name] = (xhat, history)
    return rows, (x, a.T @ b, curves)


def save(rows, example):
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = 'seed,method,iterations,restarts,time_s,objective,rel_error,mse,snr_db,support_f1'
    np.savetxt(OUT_DIR / 'prob3_trials.csv', rows, fmt='%s', delimiter=',', header=header, comments='')
    with open(OUT_DIR / 'prob3_summary.csv', 'w', encoding='utf-8') as f:
        f.write('method,metric,mean,std\n')
        for name in ('PRP+', 'GPSR-BB'):
            data = np.asarray([[float(v) for v in row[2:]] for row in rows if row[1] == name])
            for metric, mean, std in zip(header.split(',')[2:], data.mean(0), data.std(0)):
                f.write(f'{name},{metric},{mean:.8g},{std:.8g}\n')

    x, backprojection, curves = example
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, (title, values) in zip(axes, [('Original signal', x), ('Backprojection $A^Tb$', backprojection)] +
                                 [(name, curves[name][0]) for name in curves]):
        ax.plot(values, lw=0.7)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'reconstruction.png', dpi=180)
    plt.close(fig)

    for name, (_, history) in curves.items():
        plt.semilogy(history, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Original objective')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'convergence.png', dpi=180)
    plt.close()


def main():
    rows, example = [], None
    for seed in range(TRIALS):
        trial_rows, current = run(seed)
        rows.extend(trial_rows)
        example = example or current
        for row in trial_rows:
            print(f'seed={seed:2d} {row[1]:7s} iter={row[2]:4d} time={row[4]:7.2f}s '
                  f'RelErr={row[6]:.4f} SNR={row[8]:6.2f}dB F1={row[9]:.4f}')
    assert len(rows) == 2 * TRIALS and all(np.isfinite(float(v)) for row in rows for v in row[2:])
    save(rows, example)


if __name__ == '__main__':
    main()
