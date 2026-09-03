import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from utils import IMG_DIR, load_gray, psnr, run_pipeline, snr

POT, ALPHA = 'sqrt', 1e-4          # 问题一主模型势函数
BETA = 20.0                        # 正则化强度
OUT_DIR = Path('../output/prob1')
PLOT_DIR = OUT_DIR / 'plots'
LIT_DIR = OUT_DIR / 'ret'
def run(img_name, ratio, seed=0):
    x, y, amf_out, omega, u, iters, (t_det, t_opt) = run_pipeline(img_name, ratio, seed=seed,restore_kw=dict(beta=BETA, pot=POT, alpha=ALPHA, n_layers=5))
    rows = [
        ('noisy', y, psnr(y, x), snr(y, x), 0.0),
        ('amf', amf_out, psnr(amf_out, x), snr(amf_out, x), t_det),
        ('restored', u, psnr(u, x), snr(u, x), t_opt),
    ]
    print(f'=== {img_name}  noise={ratio:.0%}  seed={seed} ===')
    print(f'AMF 检出噪声像素 |Ω|={omega.sum()} ({omega.mean():.2%})，'
          f'检测 {t_det:.2f}s，优化 {iters} 次迭代 {t_opt:.2f}s')
    print(f'{"方法":<10}{"PSNR/dB":>10}{"SNR/dB":>10}')
    for name, _, p, s, _ in rows:
        print(f'{name:<10}{p:>10.2f}{s:>10.2f}')
    return rows
if __name__ == '__main__':
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
            for name, _, p, _, elapsed in rows:
                summary.append((n, r, name, p, elapsed))
            if (n, r) not in saved_grids:
                saved_grids.add((n, r))
                noisy, amf_out, restored = [row[1] for row in rows]
                x_clean = load_gray(IMG_DIR / f'{n}.tif')
                h, w = x_clean.shape
                sep = 8
                fig = np.zeros((h * 2 + sep, w * 2 + sep), float)
                fig[:h, :w] = x_clean
                fig[:h, w + sep:] = noisy
                fig[h + sep:, :w] = amf_out
                fig[h + sep:, w + sep:] = restored
                PILImage.fromarray(np.clip(fig, 0, 255).astype(np.uint8)).save(PLOT_DIR / f'{n}_r{int(r*100)}_grid.png')
    print('========== 全灰度图汇总seed=%d==========' % seed)
    print(f'{"图像":<18}{"噪声":>6}{"方法":>10}{"PSNR/dB":>10}')
    for n, r, name, p, _ in summary:
        print(f'{n:<18}{r:>6.0%}{name:>10}{p:>10.2f}')
    csv_path = LIT_DIR / 'prob1_summary.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('image,noise_ratio,method,psnr_db,time_s\n')
        for n, r, name, p, elapsed in summary:
            f.write(f'{n},{r:.2f},{name},{p:.4f},{elapsed:.4f}\n')
    print(f'\n汇总已写入: {csv_path}')
