#!/usr/bin/env python3
"""
reconstruct_lines_3d.py — 凸起樣品逐線 1D 探針去卷積 + 3D 重建（零訓練）

給定探針規格（R nm 球冠半徑、θ° 半錐角）與 Nanoscope .000 掃描檔，
對『每一條掃描線』做 1D 形態學 erosion 去卷積（還原被探針撐寬的凸起），
再把所有還原線堆疊成 2D 高度圖，最後輸出 3D 表面渲染圖（輸入 vs 還原）。

為什麼凸起好還原（相對凹孔）：
  凸起頂端會被探針準確爬過，只有『側壁被探針撐寬』(tip broadening)；
  erosion 去卷積能把撐寬收回 → 可還原區域大、結果準。
  （凹孔則因探針進不去孔底，底部資訊物理性遺失，救不回。）

逐線 1D 法（依使用者需求）：
  每條掃描線用探針的『1D 中心剖面』獨立做 grey_erosion，再堆疊成面。
  符合 AFM line-scan 的直覺、快又簡單；屬近似——忽略跨行(慢軸)的探針 2D 形狀。
  要更嚴謹可改用完整 2D（見 blind_deconvolution.py 的 reconstruct_surface）。

數學保證：erosion 還原滿足 s ≤ s_r ≤ i（s=真實表面），**不會過度去卷積**。

用法：
  python reconstruct_lines_3d.py <檔> --R 10 --theta 25 [--tip-half N] [--out 資料夾]
  例：python reconstruct_lines_3d.py datafortip/forpredict/std.000 --R 10 --theta 25

Nanoscope 解析與 Z 校正邏輯沿用 detect.py（資料長度上界、跨通道保護、Z-scale）。
"""
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (註冊 3d 投影)
from scipy.ndimage import grey_erosion, median_filter


# ══════════════════════════════════════════════════════════════════
# Nanoscope 解析（沿用 detect.py 邏輯）
# ══════════════════════════════════════════════════════════════════
def parse_nanoscope_header(fp):
    with open(fp, 'rb') as f:
        hdr = f.read(65536).decode('latin-1')

    def fv(pat, cast=str, default=None):
        m = re.search(pat, hdr)
        return cast(m.group(1)) if m else default

    n_px = fv(r'\\Samps/line:\s*(\d+)', int, 256)
    n_ln = fv(r'\\(?:Number of lines|Lines):\s*(\d+)', int, 256)
    m_sc = re.search(r'\\Scan Size:\s*([\d.]+)\s*([\d.]*)\s*(~m|nm|um)', hdr)
    scan = float(m_sc.group(1)) * 1000 if m_sc and m_sc.group(3) in ('~m', 'um') \
        else (float(m_sc.group(1)) if m_sc else 5000.)
    zss = fv(r'@Sens\. ZsensSens:\s*V\s*([\d.]+)', float, 813.1653)
    zs  = fv(r'@Sens\. Zsens:\s*V\s*([\d.]+)',     float, 32.07862)
    chs = []
    for blk in hdr.split('\\*Ciao image list'):
        if not blk.strip():
            continue
        off = re.search(r'\\Data offset:\s*(\d+)', blk)
        bpp = re.search(r'\\Bytes/pixel:\s*(\d+)', blk)
        dln = re.search(r'\\Data length:\s*(\d+)', blk)
        nm  = re.search(r'@2:Image Data:\s*S\s*\[([^\]]+)\].*?"([^"]+)"', blk)
        zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
        if off and nm:
            chs.append({
                'offset':      int(off.group(1)),
                'bpp':         int(bpp.group(1)) if bpp else 2,
                'data_length': int(dln.group(1)) if dln else None,
                'key':         nm.group(1).strip(),
                'name':        nm.group(2).strip(),
                'z_key':       zsc.group(1) if zsc else 'ZsensSens',
                'z_lsb':       float(zsc.group(2)) if zsc else 0.000375,
            })
    return {'n_px': n_px, 'n_lines': n_ln, 'scan_nm': scan, 'px_nm': scan / n_px,
            'zsens': zs, 'zsens_s': zss, 'channels': chs}


def read_height_channel(fp, meta):
    """讀取高度通道 → nm 單位 2D array（沿用 detect.py 的 Data length 上界保護）。"""
    ch_idx = 0
    for i, ch in enumerate(meta['channels']):
        if any(k.lower() in ch['name'].lower()
               for k in ['ZSensor', 'Height Sensor', 'Height']):
            ch_idx = i
            break
    ch = meta['channels'][ch_idx]
    n  = meta['n_px'] * meta['n_lines']
    dt = np.int16 if ch['bpp'] == 2 else np.int32
    read_bytes = n * ch['bpp']
    if ch.get('data_length'):
        read_bytes = min(read_bytes, ch['data_length'])   # 防跨通道污染
    with open(fp, 'rb') as f:
        f.seek(ch['offset'])
        raw = np.frombuffer(f.read(read_bytes), dtype=dt)
    actual_lines = len(raw) // meta['n_px']
    if actual_lines == 0:
        raise ValueError("通道資料不足，無法組成影像")
    if actual_lines != meta['n_lines']:
        print(f"  [注意] 實際行數 {actual_lines}（header 記 {meta['n_lines']}），以實際為準")
    img  = raw[:actual_lines * meta['n_px']].reshape(actual_lines, meta['n_px']).astype(np.float64)
    sens = meta['zsens_s'] if 'ZsensSens' in ch['z_key'] else meta['zsens']
    nm_per_lsb = ch['z_lsb'] * sens
    print(f"  [Z校正] ch='{ch['name']}'  z_lsb={ch['z_lsb']:.6g}  sens={sens:.4f}"
          f"  → {nm_per_lsb:.4f} nm/LSB")
    return img * nm_per_lsb


def despike(img, win=3, abs_thresh_nm=300.0, n_sigma=8.0):
    """去孤立尖刺（與 detect.py 同；方向無關，對凸/凹皆適用）。"""
    med   = median_filter(img, size=win)
    resid = img - med
    mad   = np.median(np.abs(resid - np.median(resid)))
    thresh = max(abs_thresh_nm, n_sigma * 1.4826 * mad)
    mask  = np.abs(resid) > thresh
    if mask.any():
        img = img.copy(); img[mask] = med[mask]
        print(f"  [去尖刺] 移除 {int(mask.sum())} 個尖刺像素（門檻 {thresh:.0f}nm）")
    return img


def flatten_rows_bumps(img, baseline_pct=10):
    """凸起樣品逐行基線校正：每行一階去傾斜，再以低百分位為表面基準（=0）。

    與孔洞版相反——凸起樣品『表面是低值、特徵向上』，故基準取 percentile(low)
    （表面），凸起在其之上為正值。
    """
    flat = img.astype(np.float64).copy()
    x = np.arange(img.shape[1])
    for i in range(img.shape[0]):
        c = np.polyfit(x, flat[i], 1)
        flat[i] -= np.polyval(c, x)
    flat -= np.percentile(flat, baseline_pct)     # 表面 → 0，凸起向上為正
    return flat


# ══════════════════════════════════════════════════════════════════
# 探針（球冠 + 直線錐壁）與逐線 1D 去卷積
# ══════════════════════════════════════════════════════════════════
def make_cone_profile_1d(half_px, px_nm, R_nm, theta_deg):
    """探針 1D 中心剖面（apex=0、向外為負）。供逐線 erosion 當 structure。"""
    r = np.abs(np.arange(-half_px, half_px + 1)) * px_nm
    th = np.radians(theta_deg)
    rt = R_nm * np.sin(th)
    prof = np.where(
        r <= rt,
        -(R_nm - np.sqrt(np.maximum(R_nm**2 - r**2, 0))),
        -(R_nm - np.sqrt(max(R_nm**2 - rt**2, 0.0))) - (np.maximum(r - rt, 0)) / np.tan(th),
    )
    prof -= prof.max()
    return prof


def auto_tip_half(img, px_nm, R_nm, theta_deg, cap=40):
    """依凸起最大高度自動估計探針視窗半徑（px），使錐壁足以涵蓋特徵側壁。

    凸起高 h 經 θ 錐探針撐寬 ≈ R·sinθ + h·tanθ；視窗需 ≥ 此側向延伸。
    """
    max_h = float(np.percentile(img, 99.5) - np.percentile(img, 10))
    r_nm  = R_nm * np.sin(np.radians(theta_deg)) + max_h * np.tan(np.radians(theta_deg))
    return int(min(cap, max(3, np.ceil(r_nm / px_nm) + 1)))


def reconstruct_per_line(img, tip_1d):
    """逐線 1D 去卷積：每條掃描線（row）獨立做 grey_erosion(line, tip_1d)。

    ⚠ 只沿 X(快軸)還原 → Y(慢軸)不被修正：被探針等向撐大的圓會變成橢圓
    （X 收窄、Y 不變）。要等向還原請用 reconstruct_2d（--mode 2d）。
    凸起經探針 dilation 變寬 → erosion 收回。滿足 s_r ≤ image（確定下界，不腦補）。
    """
    recon = np.empty_like(img)
    for i in range(img.shape[0]):
        recon[i] = grey_erosion(img[i], structure=tip_1d, mode='nearest')
    return recon


def make_cone_tip_2d(half_px, px_nm, R_nm, theta_deg):
    """探針 2D 形狀（球冠+直線錐壁，apex=0、向外為負）。供等向 2D 去卷積。"""
    size = 2 * half_px + 1
    yi, xi = np.indices((size, size))
    r = np.sqrt((yi - half_px) ** 2 + (xi - half_px) ** 2) * px_nm
    th = np.radians(theta_deg)
    rt = R_nm * np.sin(th)
    tip = np.zeros((size, size))
    m = r <= rt
    tip[m] = -(R_nm - np.sqrt(np.maximum(R_nm ** 2 - r[m] ** 2, 0)))
    zt = -(R_nm - np.sqrt(max(R_nm ** 2 - rt ** 2, 0.0)))
    tip[~m] = zt - (r[~m] - rt) / np.tan(th)
    tip -= tip.max()
    return tip


def reconstruct_2d(img, tip_2d):
    """完整 2D 去卷積：grey_erosion(img, tip_2d)。X/Y 同時還原（等向，圓不會變橢圓）。

    物理正確——探針是 2D 物體、特徵在 2D 被撐大。滿足 s_r ≤ image（不過度去卷積）。
    """
    return grey_erosion(img, structure=tip_2d, mode='nearest')


# ══════════════════════════════════════════════════════════════════
# 3D 渲染
# ══════════════════════════════════════════════════════════════════
def render_3d(panels, px_nm, out_path, suptitle):
    """panels = list of (Z_2d, title)。輸出並排 3D 表面圖（共用 z 範圍）。"""
    H, W = panels[0][0].shape
    stride = max(1, int(np.ceil(max(H, W) / 160)))      # 取樣避免 plot_surface 太慢/糊
    ys = np.arange(0, H, stride); xs = np.arange(0, W, stride)
    X, Y = np.meshgrid(xs * px_nm, ys * px_nm)
    Zs = [Z[np.ix_(ys, xs)] for Z, _ in panels]
    vmin = min(z.min() for z in Zs); vmax = max(z.max() for z in Zs)

    n = len(panels)
    fig = plt.figure(figsize=(8 * n, 7))
    for k, (Z, (_, ttl)) in enumerate(zip(Zs, panels)):
        ax = fig.add_subplot(1, n, k + 1, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='afmhot', vmin=vmin, vmax=vmax,
                        linewidth=0, antialiased=True, rcount=len(ys), ccount=len(xs))
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)'); ax.set_zlabel('z (nm)')
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=45, azim=-60)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  [3D圖] {out_path}")


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description='凸起樣品逐線 1D 探針去卷積 + 3D 重建（給定 R, θ）')
    ap.add_argument('image', help='Nanoscope 掃描檔（如 std.000）')
    ap.add_argument('--R', type=float, required=True, help='探針尖端球冠半徑 R (nm)')
    ap.add_argument('--theta', type=float, required=True, help='探針半錐角 θ (度)')
    ap.add_argument('--tip-half', type=int, default=0,
                    help='探針視窗半徑 px（0=依凸起高度自動估）')
    ap.add_argument('--mode', choices=['2d', 'line', 'both'], default='2d',
                    help="去卷積模式：2d=完整2D等向還原(推薦)；line=逐線1D(只修X→圓變橢圓)；"
                         "both=兩者並排比較")
    ap.add_argument('--baseline-pct', type=float, default=10,
                    help='表面基準百分位（凸起樣品取低百分位，預設 10）')
    ap.add_argument('--out', default='recon3d_out', help='輸出資料夾')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"讀取 Nanoscope：{args.image}")
    meta = parse_nanoscope_header(args.image)
    px_nm = meta['px_nm']
    print(f"  掃描範圍 {meta['scan_nm']:.1f} nm  解析度 {meta['n_px']}×{meta['n_lines']}"
          f"  px_nm={px_nm:.3f}")

    img = read_height_channel(args.image, meta)
    img = despike(img)
    img = flatten_rows_bumps(img, args.baseline_pct)
    print(f"  基線校正後（凸起向上為正）：[{img.min():.1f}, {img.max():.1f}] nm")

    tip_half = args.tip_half or auto_tip_half(img, px_nm, args.R, args.theta)
    print(f"\n探針：cone R={args.R}nm θ={args.theta}°  視窗半徑 {tip_half}px"
          f"（{tip_half*px_nm:.0f}nm）  模式={args.mode}")

    # 依模式計算還原（2d=等向，line=只修X→橢圓，both=兩者）
    recon_main = None
    panels = [(img, 'Input (AFM scan, tip-broadened)')]
    if args.mode in ('2d', 'both'):
        tip_2d = make_cone_tip_2d(tip_half, px_nm, args.R, args.theta)
        rec2d  = reconstruct_2d(img, tip_2d)
        panels.append((rec2d, 'Reconstructed 2D (isotropic — circle stays circle)'))
        recon_main = rec2d
        print(f"  [2D] 完整 2D 去卷積完成  下界 s_r≤i：{bool(np.all(rec2d<=img+1e-6))}")
    if args.mode in ('line', 'both'):
        tip_1d = make_cone_profile_1d(tip_half, px_nm, args.R, args.theta)
        recl   = reconstruct_per_line(img, tip_1d)
        panels.append((recl, 'Reconstructed 1D per-line (X only -> ellipse)'))
        if recon_main is None:
            recon_main = recl
        print(f"  [1D] 逐線去卷積完成（{img.shape[0]} 條線）"
              f"  下界 s_r≤i：{bool(np.all(recl<=img+1e-6))}")
        if args.mode == 'both':
            print(f"  ⚠ 注意：1D 只修 X 軸；被等向撐大的圓會在 1D 還原中變橢圓，"
                  f"請以 2D 結果為準。")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.out, f'{stem}_reconstructed.npy'), recon_main)
    np.save(os.path.join(args.out, f'{stem}_input.npy'), img)
    render_3d(panels, px_nm, os.path.join(args.out, f'{stem}_3d.png'),
              f'AFM convex reconstruction | cone R={args.R}nm theta={args.theta}deg '
              f'half={tip_half}px ({px_nm:.2f} nm/px) | mode={args.mode}')
    print(f"\n完成。結果存於：{args.out}/（_3d.png 為主結果；_reconstructed.npy = "
          f"{'2D' if args.mode!='line' else '1D'} 還原高度圖）")


if __name__ == '__main__':
    main()
