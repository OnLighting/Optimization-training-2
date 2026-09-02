import numpy as np
from PIL import Image
SMIN, SMAX = 0.0, 255.0
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
def amf_detect(y, wmax=9):
    """自适应中值滤波。返回 (输出图 out, 噪声候选掩码 omega)。

    判定规则：窗口内 zmin < y < zmax → 干净像素（保留 y）；
    否则若 zmin < zmed < zmax（中值有效）→ 判为噪声，输出 zmed；
    两条件均不满足 → 扩大窗口；到 wmax 仍未判定 → 输出最大窗口中值。
    返回的 omega：极值像素且输出 != 原值（即噪声候选集合 Ω）。
    """
    from scipy.ndimage import rank_filter
    y = y.astype(float)
    edges = list(range(3, wmax + 1, 2))            # 3,5,...,wmax
    # 中值排序位 = w*w//2（奇数窗口下恰为中间），rank 必须传 int
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
    if name == 'sqrt':          # sqrt(t^2 + alpha)
        return (lambda t: np.sqrt(t * t + alpha),
                lambda t: t / np.sqrt(t * t + alpha))
    if name == 'power':         # |t|^alpha, 1 < alpha <= 2
        def dphi(t):
            return np.sign(t) * alpha * np.abs(t) ** (alpha - 1)
        return lambda t: np.abs(t) ** alpha, dphi
    if name == 'logcosh':       # log(cosh(alpha t)) / alpha
        a = alpha
        def phi(t):
            at = a * np.abs(t)
            return (at + np.log1p(np.exp(-2 * at)) - np.log(2.0)) / a
        return phi, (lambda t: np.tanh(a * t))
    if name == 'log':           # |t|/alpha - log(1 + |t|/alpha)
        def phi(t):
            at = np.abs(t) / alpha
            return at - np.log1p(at)
        def dphi(t):
            at = np.abs(t)
            return np.sign(t) * at / (alpha * (alpha + at))
        return phi, dphi
    if name == 'huber':         # Huber
        def phi(t):
            at = np.abs(t)
            return np.where(at <= alpha, t * t / (2 * alpha), at - alpha / 2)
        return phi, (lambda t: np.clip(t / alpha, -1.0, 1.0))
    return None
def smooth_abs(mu):
    return (lambda t: np.sqrt(t * t + mu * mu),
            lambda t: t / np.sqrt(t * t + mu * mu))
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
    edges = _edges(y.shape)
    def v_image(u):
        v = u.copy()
        v[~omega] = y[~omega]
        return v
    def objective(u):
        v = v_image(u)
        val = psi(u[omega] - y[omega]).sum()
        for (pr, pc), (qr, qc) in edges:
            diff = u[pr, pc] - v[qr, qc]
            w = np.where(omega[qr, qc], 1.0, 2.0)      # 跨边界边 2×
            val += 0.5 * beta * (w * phi(diff) * omega[pr, pc]).sum()
        return val
    def gradient(u):
        v = v_image(u)
        g = np.zeros_like(u)
        g[omega] = dpsi(u[omega] - y[omega])
        for (pr, pc), (qr, qc) in edges:
            add = np.zeros_like(u)
            add[pr, pc] = beta * dphi(u[pr, pc] - v[qr, qc])
            g += add * omega
        return g
    return objective, gradient
# 逐元素截断到 [SMIN, SMAX]
def project(u):
    return np.clip(u, SMIN, SMAX)
def mse(xhat, x):
    return np.mean((xhat - x) ** 2)
def psnr(xhat, x):
    return 10 * np.log10(SMAX ** 2 / mse(xhat, x))
def snr(xhat, x):
    return 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(xhat - x))