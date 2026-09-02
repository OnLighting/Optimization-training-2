import time
from pathlib import Path
import numpy as np
from PIL import Image
from utils import add_salt_pepper_noise, amf_detect, load_gray, make_potential, make_problem, project, psnr, snr

IMG_DIR = Path('../参考文献及测试图片/standard_test_images')
POT, ALPHA = 'sqrt', 1e-4          # 问题一主模型势函数
BETA = 20.0                        # 正则化强度
MU0, MU_MIN, RHO = 1.0, 1e-3, 0.1 # 光滑参数延拓：mu_{j+1} = rho * mu_j
MAXIT, TOL = 80, 1e-4              # 每层最大迭代 / 投影梯度残差阈值
ETAMIN, ETAMAX = 1e-10, 1e10       # BB 步长截断
# ponytail: 当前每图跑 5 层延拓 × 200 步 ≈ 400s；改 Lipschitz 单调回溯 + 缩短延拓层数
#           可压到 30s 内；AMF Python 循环另外 5s 改 scipy.ndimage.rank_filters
OUT_DIR = Path('../output/prob1')
PLOT_DIR = OUT_DIR / 'plots'
LIT_DIR = OUT_DIR / 'ret'
def projected_gradient(f, g, u0, maxit=MAXIT, tol=TOL):
    u = project(u0)
    fu, gu = f(u), g(u)
    for k in range(1, maxit + 1):
        if k >= 2:
            s, z = u - u_prev, gu - g_prev
            sz = (s * z).sum()
            eta = (s * s).sum() / sz if sz > 1e-30 else 1.0
        else:
            eta = 1.0
        eta = float(np.clip(eta / 5.0, ETAMIN, ETAMAX))
        alpha, c1, decay = 1.0, 1e-4, 0.5
        while alpha > 1e-20:
            un = project(u - alpha * gu)
            if f(un) <= fu + c1 * alpha * (gu * (un - u)).sum():
                break
            alpha *= decay
        else:
            un = project(u - gu)     # 保险：步长耗尽时退回单位步长
        fn, gn = f(un), g(un)
        u_prev, g_prev, u, fu, gu = u, gu, un, fn, gn
    return u, k
# 光滑延拓投影梯度恢复
def restore(y, omega, beta=BETA, pot=POT, alpha=ALPHA):
    phi, dphi = make_potential(pot, alpha)
    f, _ = make_problem(y, omega, beta, phi, dphi, mu=MU0)
    u = np.where(omega, y, y)        # 初值：AMF 输出
    total = 0
    mu = MU0
    while True:
        f, g = make_problem(y, omega, beta, phi, dphi, mu=mu)
        u, k = projected_gradient(f, g, u)
        total += k
        if mu <= MU_MIN:
            break
        mu *= RHO
    return u, total
def run(img_name, ratio, seed=0):
    rng = np.random.default_rng(seed)
    x = load_gray(IMG_DIR / f'{img_name}.tif')
    y = add_salt_pepper_noise(x, ratio, rng)
    t0 = time.perf_counter()
    amf_out, omega = amf_detect(y, wmax=9 if ratio >= 0.4 else 7)
    t1 = time.perf_counter()
    u, iters = restore(y, omega)
    t2 = time.perf_counter()
    rows = [
        ('noisy', y, psnr(y, x), snr(y, x)),
        ('amf', amf_out, psnr(amf_out, x), snr(amf_out, x)),
        ('restored', u, psnr(u, x), snr(u, x)),
    ]
    print(f'\n=== {img_name}  noise={ratio:.0%}  seed={seed} ===')
    print(f'AMF 检出噪声像素 |Ω|={omega.sum()} ({omega.mean():.2%})，'
          f'检测 {t1-t0:.2f}s，优化 {iters} 次迭代 {t2-t1:.2f}s')
    print(f'{"方法":<10}{"PSNR/dB":>10}{"SNR/dB":>10}')
    for name, _, p, s in rows:
        print(f'{name:<10}{p:>10.2f}{s:>10.2f}')
    for name, img, _, _ in rows:
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(PLOT_DIR / f'{img_name}_r{int(ratio*100)}_{name}.png')
    return rows
if __name__ == '__main__':
    import sys
    from PIL import Image as PILImage
    LIT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = sorted(p.stem for p in IMG_DIR.glob('*.tif'))
    print(f"图片共{len(names)}张")
    seed = 0
    summary = []                              # 每图每噪声率各方法 PSNR
    saved_grids = set()                       # 每图每噪声率只保存一张
    for n in names:
        for r in (0.3, 0.5):
            rows = run(n, r, seed=seed)
            for name, _, p, _ in rows:
                summary.append((n, r, name, p))
            if (n, r) not in saved_grids:
                saved_grids.add((n, r))
                amf_out = np.asarray(PILImage.open(PLOT_DIR/ f'{n}_r{int(r*100)}_amf.png'), float)
                noisy = np.asarray(PILImage.open(PLOT_DIR / f'{n}_r{int(r*100)}_noisy.png'), float)
                restored = np.asarray(PILImage.open(PLOT_DIR / f'{n}_r{int(r*100)}_restored.png'), float)
                x_clean = load_gray(IMG_DIR / f'{n}.tif')
                h, w = x_clean.shape
                sep = 8
                fig = np.zeros((h * 2 + sep, w * 2 + sep), float)
                fig[:h, :w] = x_clean
                fig[:h, w + sep:] = noisy
                fig[h + sep:, :w] = amf_out
                fig[h + sep:, w + sep:] = restored
                PILImage.fromarray(np.clip(fig, 0, 255).astype(np.uint8)).save(PLOT_DIR / f'{n}_r{int(r*100)}_grid.png')
    print('\n\n========== 全灰度图汇总seed=%d==========' % seed)
    print(f'{"图像":<18}{"噪声":>6}{"方法":>10}{"PSNR/dB":>10}')
    for n, r, name, p in summary:
        print(f'{n:<18}{r:>6.0%}{name:>10}{p:>10.2f}')
    csv_path = LIT_DIR / 'prob1_summary.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('image,noise_ratio,method,psnr_db\n')
        for n, r, name, p in summary:
            f.write(f'{n},{r:.2f},{name},{p:.4f}\n')
    print(f'\n汇总已写入: {csv_path}')