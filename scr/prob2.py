from pathlib import Path
from utils import psnr, run_pipeline, snr
import numpy as np
from utils import IMG_DIR, load_gray, make_potential
OUT_DIR = Path('../output/prob2')
BETA_GRID = [5.0, 10.0, 20.0, 30.0, 50.0]
CALIB_KW = dict(n_layers=2, maxit=25, tol=1e-3)
NOISE_RATIOS = (0.3, 0.5)
SEED = 0
IMAGE_NAMES = sorted(p.stem for p in IMG_DIR.glob('*.tif'))
VAL_NAMES = IMAGE_NAMES[:3]
TEST_NAMES = IMAGE_NAMES[3:]
POTENTIALS = ['sqrt', 'power', 'logcosh', 'log', 'huber']
ALPHA_FIXED = {'sqrt': 1e-06, 'power': 1.1, 'logcosh': 2.5, 'log': 0.9, 'huber': 1.0}
T_FAR = 10
CANDIDATES = {
    'sqrt':    [1e-6, 1e-4, 1e-2, 1.0],
    'power':   [1.1,1.2, 1.3, 1.5, 1.8],
    'logcosh': [1.0, 1.5, 2.0, 2.5],
    'log':     [0.8,0.9, 1.0, 1.3],
    'huber':   [1.0, 1.3, 1.5, 2],
}
def edge_samples(img_name):
    x = load_gray(IMG_DIR / f'{img_name}.tif')
    d = np.concatenate([np.abs(np.diff(x, axis=0)).ravel(),np.abs(np.diff(x, axis=1)).ravel()])
    return d, d[d >= T_FAR]
def explore_alpha(img_name):
    d, t = edge_samples(img_name)
    print(f'=== {img_name}  差分总数 {d.size}，边缘区 |t|>={T_FAR:.0f}: '
          f'{t.size} 个 ({t.size / d.size:.1%})，区间 [{t.min():.0f}, {t.max():.0f}]，'
          f'中位数 {np.median(t):.0f} ===')
    print(f'{"势函数":<9}{"alpha":>8} {"rel均值":>8}{"rel最大":>8}'
          f'{"斜偏均值":>9}{"斜偏最大":>9}{"数值校验":>10}  结论')
    for pot, cands in CANDIDATES.items():
        for a in cands:
            phi, dphi = make_potential(pot, a)
            rel = np.abs(phi(t) - t) / t
            ddev = np.abs(dphi(t) - 1.0)
            h = 1e-3 * np.maximum(t, 1.0)
            num = np.abs(dphi(t) - (phi(t + h) - phi(t - h)) / (2 * h))
            mark = '*' if a == ALPHA_FIXED[pot] else ' '
            verdict = ('≈|t|' if rel.mean() < 0.1 and ddev.mean() < 0.1 else
                       '接近' if rel.mean() < 0.5 else '偏离')
            print(f'{pot:<9}{a:>8}{mark}{rel.mean():>9.3f}{rel.max():>9.3f}'
                  f'{ddev.mean():>10.3f}{ddev.max():>10.3f}{num.max():>12.1e}  {verdict}')
        print()
def psnr_batch(pot, alpha, beta, names, ratio):
    psnrs = []
    for n in names:
        x, y, _, omega, u, _, _ = run_pipeline(
            n, ratio, seed=SEED,
            restore_kw=dict(beta=beta, pot=pot, alpha=alpha, **CALIB_KW))
        psnrs.append(psnr(np.where(omega, u, y), x))
    return float(np.mean(psnrs))
# 对单一 pot 在单噪声率上做beta搜索。
def find_best_beta(pot, ratio):
    best_v, best_b = -np.inf, BETA_GRID[0]
    for b in BETA_GRID:
        v = psnr_batch(pot, ALPHA_FIXED[pot], b, VAL_NAMES, ratio)
        if v > best_v:
            best_v, best_b = v, b
    return ALPHA_FIXED[pot], best_b, best_v
def run_one(img_name, ratio, pot, alpha, beta):
    x, y, _, omega, u, iters, (t_det, t_opt) = run_pipeline(
        img_name, ratio, seed=SEED,
        restore_kw=dict(beta=beta, pot=pot, alpha=alpha))
    xhat = np.where(omega, u, y)
    return dict(name=img_name, ratio=ratio, pot=pot, alpha=alpha,
                beta=beta, iters=iters, t_det=t_det, t_opt=t_opt,
                psnr=psnr(xhat, x), snr=snr(xhat, x))
def main():
    OUT_DIR.mkdir(exist_ok=True)
    # print('===== alpha 参数标定 =====')
    # explore_alpha('lena_gray_512')
    # print(f"最终alpha{ALPHA_FIXED}")
    calib = {}
    print('===== beta 参数标定 =====')
    for pot in POTENTIALS:
        calib[pot] = {}
        for r in NOISE_RATIOS:
            a, b, v = find_best_beta(pot, r)
            calib[pot][r] = (a, b)
            print(f'pot={pot:<8} ratio={r:.0%}  alpha={a}  beta={b}  val_psnr={v:.2f}')
    rows = []
    print(f'{"图像":<18}{"噪声":>6}{"势函数":>10}{"alpha":>10}{"beta":>6}'
          f'{"迭代":>6}{"时间/s":>8}{"PSNR/dB":>10}{"SNR/dB":>10}')
    for n in TEST_NAMES:
        for r in NOISE_RATIOS:
            for pot in POTENTIALS:
                a, b = calib[pot][r]
                rec = run_one(n, r, pot, a, b)
                rows.append(rec)
                print(f'{rec["name"]:<18}{rec["ratio"]:>6.0%}{rec["pot"]:>10}'
                      f'{rec["alpha"]:>10}{rec["beta"]:>6.0f}'
                      f'{rec["iters"]:>6d}{rec["t_opt"]:>8.2f}'
                      f'{rec["psnr"]:>10.2f}{rec["snr"]:>10.2f}')
    # 3) 按噪声率和势函数汇总主测试集
    print('\n===== 各势函数在主测试集上的均值和标准差 =====')
    print(f'{"势函数":<10}{"噪声":>6}{"mean PSNR":>12}{"std PSNR":>12}'
          f'{"mean SNR":>12}{"std SNR":>12}{"mean iter":>12}{"std iter":>12}'
          f'{"mean t/s":>12}{"std t/s":>12}')
    summary = []
    for r in NOISE_RATIOS:
        for pot in POTENTIALS:
            sub = [x for x in rows if x['pot'] == pot and x['ratio'] == r]
            ps = np.array([x['psnr'] for x in sub])
            ss = np.array([x['snr'] for x in sub])
            it = np.array([x['iters'] for x in sub])
            tt = np.array([x['t_opt'] for x in sub])
            print(f'{pot:<10}{r:>6.0%}{ps.mean():>12.2f}{ps.std():>12.2f}'
                  f'{ss.mean():>12.2f}{ss.std():>12.2f}'
                  f'{it.mean():>12.1f}{it.std():>12.1f}'
                  f'{tt.mean():>12.2f}{tt.std():>12.2f}')
            summary.append((pot, r, ps.mean(), ps.std(), ss.mean(), ss.std(),it.mean(), it.std(), tt.mean(), tt.std()))
    # 4) 落盘
    csv = OUT_DIR / 'prob2_summary.csv'
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('potential,noise_ratio,alpha,beta,mean_psnr,std_psnr,'
                'mean_snr,std_snr,mean_iter,std_iter,mean_time_s,std_time_s\n')
        for pot, r, mp, sp, ms, ss, mi, si, mt, st in summary:
            a, b = calib[pot][r]
            f.write(f'{pot},{r:.2f},{a},{b:.4f},{mp:.4f},{sp:.4f},'
                    f'{ms:.4f},{ss:.4f},{mi:.2f},{si:.2f},{mt:.4f},{st:.4f}\n')
    print(f'\n汇总写入: {csv}')
if __name__ == '__main__':
    main()
