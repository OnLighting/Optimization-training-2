import time
import logging
from pathlib import Path
import numpy as np
from PIL import Image
logging.getLogger('tifffile').addFilter(
    lambda record: 'not a valid EXTRASAMPLE' not in record.getMessage())
SMIN, SMAX = 0.0, 255.0
IMG_DIR = Path('../参考文献及测试图片/standard_test_images')
def load_gray(path):
    try:
        return np.asarray(Image.open(path).convert('L'), float)
    except Exception:
        import tifffile
        a = tifffile.imread(path)
        if a.ndim == 3:
            a = a[..., 0]
        return a.astype(float)
#椒盐噪声生成，对称：p = q = r/2
def add_salt_pepper_noise(img, ratio, rng):
    y = img.astype(float).copy()
    r = rng.random(y.shape)
    y[r < ratio / 2] = SMIN
    y[r > 1 - ratio / 2] = SMAX
    return y
# 自适应中值滤波。返回 (输出图 out, 噪声候选掩码 omega)
def amf_detect(y, wmax=9):
    from scipy.ndimage import rank_filter
    y = y.astype(float)
    edges = list(range(3, wmax + 1, 2))            # 3,5,...,wmax
    zmed_layer = [rank_filter(y, size=w, rank=w * w // 2) for w in edges]
    zmin_layer = [rank_filter(y, size=w, rank=0) for w in edges]
    zmax_layer = [rank_filter(y, size=w, rank=w * w - 1) for w in edges]
    out = y.copy()
    done = np.zeros(y.shape, bool)
    for w, zmed, zmin, zmax in zip(edges, zmed_layer, zmin_layer, zmax_layer):
        not_extreme = (zmin < y) & (y < zmax)       # 干净像素
        replaceable = (~not_extreme) & (zmin < zmed) & (zmed < zmax)
        decide = ~done & (not_extreme | replaceable)
        out[replaceable] = zmed[replaceable]
        done |= decide
        if done.all():
            break
    todo = ~done
    if todo.any():
        out[todo] = zmed_layer[-1][todo]             # 最大窗口中值兜底
    omega = ((y == SMIN) | (y == SMAX)) & (out != y)
    return out, omega
# 势函数及导数
def make_potential(name, alpha):
    # sqrt(t^2 + alpha)
    if name == 'sqrt':
        return (lambda t: np.sqrt(t * t + alpha),
                lambda t: t / np.sqrt(t * t + alpha))
    # |t|^alpha, 1 < alpha <= 2
    if name == 'power':
        def dphi(t):
            return np.sign(t) * alpha * np.abs(t) ** (alpha - 1)
        return lambda t: np.abs(t) ** alpha, dphi
    # log(cosh(alpha t)) / alpha
    if name == 'logcosh':
        a = alpha
        def phi(t):
            at = a * np.abs(t)
            return (at + np.log1p(np.exp(-2 * at)) - np.log(2.0)) / a
        return phi, (lambda t: np.tanh(a * t))
    # |t|/alpha - log(1 + |t|/alpha)
    if name == 'log':
        def phi(t):
            at = np.abs(t) / alpha
            return at - np.log1p(at)
        def dphi(t):
            at = np.abs(t)
            return np.sign(t) * at / (alpha * (alpha + at))
        return phi, dphi
    # Huber
    if name == 'huber':
        def phi(t):
            at = np.abs(t)
            return np.where(at <= alpha, t * t / (2 * alpha), at - alpha / 2)
        return phi, (lambda t: np.clip(t / alpha, -1.0, 1.0))
    return None
def smooth_abs(mu):
    return lambda t: np.sqrt(t * t + mu * mu), lambda t: t / np.sqrt(t * t + mu * mu)
# 四邻域无向边
def _edges(shape):
    m, n = shape
    return [
        ((slice(1, m), slice(0, n)), (slice(0, m - 1), slice(0, n))),   # 上邻 q=(i-1,j)
        ((slice(0, m - 1), slice(0, n)), (slice(1, m), slice(0, n))),   # 下邻 q=(i+1,j)
        ((slice(0, m), slice(1, n)), (slice(0, m), slice(0, n - 1))),   # 左邻 q=(i,j-1)
        ((slice(0, m), slice(0, n - 1)), (slice(0, m), slice(1, n))),   # 右邻 q=(i,j+1)
    ]
# 构造文献 [3] 简化泛函 G 的光滑化版本
def make_problem(y, omega, beta, phi, dphi, mu):
    psi, dpsi = smooth_abs(mu)
    edges = [(p, q, np.where(omega[q], 1.0, 2.0)) for p, q in _edges(y.shape)]
    def v_image(u):
        v = u.copy()
        v[~omega] = y[~omega]
        return v
    def objective(u):
        v = v_image(u)
        val = psi(u[omega] - y[omega]).sum()
        for (pr, pc), (qr, qc), w in edges:
            diff = u[pr, pc] - v[qr, qc]
            val += 0.5 * beta * (w * phi(diff) * omega[pr, pc]).sum()
        return val
    def gradient(u):
        v = v_image(u)
        g = np.zeros_like(u)
        g[omega] = dpsi(u[omega] - y[omega])
        for (pr, pc), (qr, qc), _ in edges:
            g[pr, pc] += beta * dphi(u[pr, pc] - v[qr, qc]) * omega[pr, pc]
        return g
    return objective, gradient
# 逐元素截断到 [SMIN, SMAX]
def project(u):
    return np.clip(u, SMIN, SMAX)
# BB 步长 + Armijo 线搜索的投影梯度
def projected_gradient(f, g, u0, maxit=80, tol=1e-4):
    u = project(u0)
    fu, gu = f(u), g(u)
    u_prev, g_prev = u.copy(), gu.copy()
    for k in range(1, maxit + 1):
        residual = np.max(np.abs(u - project(u - gu))) / max(1.0, np.max(np.abs(u)))
        if residual <= tol:
            return u, k - 1
        if k >= 2:
            s = u - u_prev
            z = gu - g_prev
            sz = (s * z).sum()
            eta = (s * s).sum() / sz if sz > 1e-30 else 1.0
        else:
            eta = 1.0
        step = float(np.clip(eta, 1e-10, 1e10))
        c1, decay = 1e-4, 0.5
        while step > 1e-20:
            un = project(u - step * gu)
            fn = f(un)
            if fn <= fu + c1 * (gu * (un - u)).sum():
                break
            step *= decay
        else:
            return u, k - 1
        gn = g(un)
        u_prev, g_prev, u, fu, gu = u, gu, un, fn, gn
    return u, k
# 光滑延拓mu_j = mu0 * rho^j，共 n_layers 层, 投影梯度恢复
def restore(y, omega, beta=20.0, pot='sqrt', alpha=1e-4, mu0=1.0, rho=0.1,
            n_layers=4, maxit=80, tol=1e-4):
    phi, dphi = make_potential(pot, alpha)
    u = y.copy()
    total = 0
    mu = mu0
    for _ in range(n_layers):
        f, g = make_problem(y, omega, beta, phi, dphi, mu=mu)
        u, k = projected_gradient(f, g, u, maxit=maxit, tol=tol)
        total += k
        mu *= rho
    return u, total
# 加载 → 加噪 → AMF 检测 → 恢复，带分段计时
def run_pipeline(img_name, ratio, seed=0, restore_kw=None, wmax=None):
    rng = np.random.default_rng(seed)
    x = load_gray(IMG_DIR / f'{img_name}.tif')
    y = add_salt_pepper_noise(x, ratio, rng)
    t0 = time.perf_counter()
    amf_out, omega = amf_detect(y, wmax=wmax or (9 if ratio >= 0.4 else 7))
    t1 = time.perf_counter()
    if restore_kw is None:
        restore_kw = {}
    u, iters = restore(y, omega, **restore_kw)
    t2 = time.perf_counter()
    return x, y, amf_out, omega, u, iters, (t1 - t0, t2 - t1)
def mse(xhat, x):
    return np.mean((xhat - x) ** 2)
def psnr(xhat, x):
    return 10 * np.log10(SMAX ** 2 / mse(xhat, x))
def snr(xhat, x):
    return 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(xhat - x))
