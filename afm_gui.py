"""
AFM 盲探針重建工具 v4.2
修正清單（相對 v4.1）：
  [FIX-9]  Sidebar 無法向下捲動 → 改用 Canvas+Scrollbar 捲動容器，支援滑鼠滾輪
  [FIX-10] Cone model 在 cross-section 往上長 → 繪製前對齊基線（令 cone.max()=0）
  [FIX-11] flatten_rows percentile(80) 在孔洞占多數時基線反落在孔內 → 改用 percentile(95)
           同時新增 shape_type 參數：孔洞型用高百分位(95)，柱體型用低百分位(5)

依賴：pip install numpy scipy matplotlib Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, re, platform
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# ── Matplotlib CJK 字型（讓圖中中文正確顯示）────────────────────────
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'Microsoft YaHei',   # Windows
    'PingFang TC', 'PingFang SC', 'Heiti TC',  # macOS
    'Noto Sans CJK TC', 'Noto Sans CJK SC',    # Linux
    'WenQuanYi Micro Hei', 'SimHei',
] + matplotlib.rcParams.get('font.sans-serif', [])
matplotlib.rcParams['axes.unicode_minus'] = False  # 負號不用方塊替代
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import label, center_of_mass, shift, gaussian_filter
from scipy.io import savemat
from scipy.signal import fftconvolve  # [FIX-5] 取代 correlate2d

C = {
    'bg':'#0f1117','panel':'#1a1d27','border':'#2a2d3a',
    'accent':'#4f9cf9','accent2':'#7c3aed','success':'#22c55e','warn':'#f59e0b','danger':'#ef4444',
    'text':'#e2e8f0','muted':'#64748b','input_bg':'#252836','hover':'#2d3348',
}

# ── 跨平台字型設定 ────────────────────────────────────────────────
def get_fonts():
    sys = platform.system()
    if sys == 'Windows':
        cjk = 'Microsoft JhengHei'
        try:
            tk.font = __import__('tkinter.font', fromlist=['font'])
            tk.font.Font(family=cjk)
        except Exception:
            cjk = 'Microsoft YaHei'
    elif sys == 'Darwin':
        cjk = 'PingFang TC'
    else:
        cjk = 'Noto Sans CJK TC'
    mono = 'Courier New' if platform.system() == 'Windows' else 'Courier'
    return {
        'normal' : (cjk, 10),
        'small'  : (cjk,  9),
        'tiny'   : (cjk,  8),
        'bold'   : (cjk, 10, 'bold'),
        'title'  : (cjk, 20, 'bold'),
        'sub'    : (cjk, 10),
        'mono'   : (mono, 8),
        'mono_n' : (mono, 10),
    }

# ══════════════════════════════════════════════════════════════════
# Float Entry
# ══════════════════════════════════════════════════════════════════
def make_float_entry(parent, textvariable, font=None, **kwargs):
    def validate(val):
        if val in ('', '-', '.', '-.'): return True
        try: float(val); return True
        except ValueError:
            return val.count('.') <= 1 and all(c in '0123456789.-' for c in val)
    vcmd = (parent.register(validate), '%P')
    if font is None:
        sys = platform.system()
        if sys == 'Windows':   font = ('Microsoft JhengHei', 10)
        elif sys == 'Darwin':  font = ('PingFang TC', 10)
        else:                  font = ('Noto Sans CJK TC', 10)
    return tk.Entry(parent, textvariable=textvariable,
                    validate='key', validatecommand=vcmd,
                    bg=C['input_bg'], fg=C['text'],
                    insertbackground=C['text'], relief='flat',
                    font=font, bd=0, **kwargs)

# ══════════════════════════════════════════════════════════════════
# Nanoscope 讀取
# ══════════════════════════════════════════════════════════════════
def parse_nanoscope(fp):
    with open(fp, 'rb') as f:
        raw = f.read(65536)
    hdr = raw.decode('latin-1')

    def fv(pat, cast=str, default=None):
        m = re.search(pat, hdr)
        return cast(m.group(1)) if m else default

    n_px = fv(r'\\Samps/line:\s*(\d+)', int, 256)
    n_ln = fv(r'\\(?:Number of lines|Lines):\s*(\d+)', int, 256)
    m = re.search(r'\\Scan Size:\s*([\d.]+)\s*([\d.]*)\s*(~m|nm|um)', hdr)
    scan = float(m.group(1)) * 1000 if m and m.group(3) in ('~m', 'um') \
           else (float(m.group(1)) if m else 5000.)
    zss = fv(r'@Sens\. ZsensSens:\s*V\s*([\d.]+)', float, 813.1653)
    zs  = fv(r'@Sens\. Zsens:\s*V\s*([\d.]+)',     float, 32.07862)
    chs = []
    for blk in hdr.split('\\*Ciao image list'):
        if not blk.strip(): continue
        off = re.search(r'\\Data offset:\s*(\d+)', blk)
        bpp = re.search(r'\\Bytes/pixel:\s*(\d+)', blk)
        nm  = re.search(r'@2:Image Data:\s*S\s*\[([^\]]+)\].*?"([^"]+)"', blk)
        zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
        if off and nm:
            chs.append({
                'offset': int(off.group(1)),
                'bpp':    int(bpp.group(1)) if bpp else 2,
                'key':    nm.group(1).strip(),
                'name':   nm.group(2).strip(),
                'z_key':  zsc.group(1) if zsc else 'ZsensSens',
                'z_lsb':  float(zsc.group(2)) if zsc else 0.000375,
            })
    return {
        'n_px': n_px, 'n_lines': n_ln, 'scan_nm': scan, 'px_nm': scan / n_px,
        'zsens': zs, 'zsens_s': zss, 'channels': chs, 'filepath': fp,
    }

def read_channel(fp, meta, idx=0):
    ch = meta['channels'][idx]
    n  = meta['n_px'] * meta['n_lines']
    dt = np.int16 if ch['bpp'] == 2 else np.int32
    with open(fp, 'rb') as f:
        f.seek(ch['offset'])
        raw = np.frombuffer(f.read(n * ch['bpp']), dtype=dt)
    img  = raw.reshape(meta['n_lines'], meta['n_px']).astype(np.float64)
    sens = meta['zsens_s'] if 'ZsensSens' in ch['z_key'] else meta['zsens']
    return img * ch['z_lsb'] * sens

# [FIX-11] 自適應基線：
#   孔洞型用 percentile(95)，表面像素是高值；掃描範圍<pitch 時孔洞佔多數，
#   percentile(80) 可能落在孔內，必須取更高百分位才能命中表面。
#   柱體型用 percentile(5)，基底像素是低值。
def flatten_rows(img, shape_type='trapezoid_hole'):
    flat = img.copy()
    x    = np.arange(img.shape[1])
    for i in range(img.shape[0]):
        c = np.polyfit(x, flat[i], 1)
        flat[i] -= np.polyval(c, x)
    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        baseline = np.percentile(flat, 95)   # 孔洞型：表面是高值
    else:
        baseline = np.percentile(flat, 5)    # 柱體型：基底是低值
    return flat - baseline

def auto_height_ch(meta):
    for p in ['ZSensor', 'Height Sensor', 'Height']:
        for i, ch in enumerate(meta['channels']):
            if p.lower() in ch['name'].lower():
                return i
    return 0

# ══════════════════════════════════════════════════════════════════
# 自動計算合理參數
# ══════════════════════════════════════════════════════════════════
def auto_params(geom, shape_type, meta):
    """
    根據幾何與掃描參數自動推算：
    - half     : patch 半徑（px）
    - max_sz   : 最大連通區域面積（px²）
    - pct_hi/lo: 偵測閾值百分位數
    - warnings : 警告訊息列表
    """
    px = meta['px_nm']
    n  = meta['n_px']
    sc = meta['scan_nm']
    p  = geom['pitch']

    r_max_nm = max(geom['d_bot'], geom['d_top']) / 2
    r_max_px = r_max_nm / px

    half = max(int(r_max_px * 1.2) + 5, 15)
    half = min(half, n // 4 - 2)

    feat_area = np.pi * r_max_px ** 2
    max_sz    = int(feat_area * 3) + 100
    max_sz    = max(max_sz, 500)

    feat_fill = feat_area / (n * n)
    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        pct = max(5, min(15, int(feat_fill * 200)))
        pct_lo, pct_hi = pct, None
    else:
        pct = max(70, min(95, 100 - int(feat_fill * 150)))
        pct_lo, pct_hi = None, pct

    warnings = []
    if p > sc:
        warnings.append(f'⚠ 週期({p}nm) > 掃描範圍({sc:.0f}nm)，視野內約 {sc/p:.1f} 個特徵')
    if half >= n // 4:
        warnings.append(f'⚠ Patch 半徑({half}px)已達上限，建議縮小掃描範圍或降低 pitch')
    if feat_fill > 0.3:
        warnings.append(f'⚠ 特徵占影像 {feat_fill*100:.0f}%，建議用更大掃描範圍')

    return {
        'half': half, 'max_sz': max_sz,
        'pct_lo': pct_lo, 'pct_hi': pct_hi,
        'r_max_px': r_max_px, 'feat_area': feat_area,
        'warnings': warnings,
    }

# ══════════════════════════════════════════════════════════════════
# GT 生成 [FIX-8] 向量化 modulo 取代雙層 for loop
# ══════════════════════════════════════════════════════════════════
def make_gt(n_px, scan_nm, geom, shape_type):
    px    = scan_nm / n_px
    p_px  = geom['pitch'] / px      # pitch in pixels
    h     = geom['height']
    r_bot = (geom['d_bot'] / 2) / px
    r_top = (geom['d_top'] / 2) / px

    yi, xi = np.indices((n_px, n_px))
    # 對週期取餘數，計算每個 pixel 到最近特徵中心的距離（向量化）
    # 偏移 p_px/2 使中心落在 [0, p_px) 中央
    cx_rel = (xi % p_px) - p_px / 2
    cy_rel = (yi % p_px) - p_px / 2
    dist   = np.sqrt(cx_rel ** 2 + cy_rel ** 2)

    if shape_type == 'cylinder_hole':
        gt = np.full((n_px, n_px), h)
        gt[dist <= r_bot] = 0.

    elif shape_type == 'trapezoid_hole':
        gt = np.full((n_px, n_px), h)
        side = (dist > r_bot) & (dist <= r_top)
        if r_top > r_bot:
            gt[side] = h * (dist[side] - r_bot) / (r_top - r_bot)
        gt[dist <= r_bot] = 0.

    else:   # trapezoid_pillar
        gt    = np.zeros((n_px, n_px))
        r_min = min(r_bot, r_top)
        r_max = max(r_bot, r_top)
        gt[dist <= r_top] = h
        side = (dist > r_min) & (dist <= r_max)
        if r_max > r_min:
            np.maximum(gt[side],
                       h * (r_max - dist[side]) / (r_max - r_min),
                       out=gt[side])

    return gt

def make_gt_patch(geom, shape_type, half, px_nm, blur_sigma=1.5):
    """單特徵理想 GT patch，邊緣高斯模糊消除銳利切斷假影。"""
    size  = 2 * half
    gt    = np.zeros((size, size))
    yi, xi = np.indices((size, size))
    dist  = np.sqrt((xi - half) ** 2 + (yi - half) ** 2)
    r_bot = (geom['d_bot'] / 2) / px_nm
    r_top = (geom['d_top'] / 2) / px_nm
    h     = geom['height']

    if shape_type == 'cylinder_hole':
        gt[dist <= r_bot] = -h
    elif shape_type == 'trapezoid_hole':
        gt[dist <= r_bot] = -h
        side = (dist > r_bot) & (dist <= r_top)
        if r_top > r_bot:
            gt[side] = -h * (r_top - dist[side]) / (r_top - r_bot)
    else:
        r_min = min(r_bot, r_top)
        r_max = max(r_bot, r_top)
        gt[dist <= r_top] = h
        side = (dist > r_min) & (dist <= r_max)
        if r_max > r_min:
            gt[side] = h * (r_max - dist[side]) / (r_max - r_min)

    if blur_sigma > 0:
        gt = gaussian_filter(gt, sigma=blur_sigma)
    return gt

# [FIX-5] 用 fftconvolve 取代 correlate2d，大影像快 10–50×
def align_gt(scan, gt, shape_type):
    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        sb = (scan < np.percentile(scan, 15)).astype(float)
        gb = (gt < gt.max() * 0.1).astype(float)
    else:
        sb = (scan > np.percentile(scan, 80)).astype(float)
        gb = (gt > gt.max() * 0.5).astype(float)
    # fftconvolve(sb, gb[::-1,::-1]) 等效於 correlate2d(sb, gb)
    c  = fftconvolve(sb, gb[::-1, ::-1], mode='same')
    pk = np.unravel_index(c.argmax(), c.shape)
    dy = pk[0] - c.shape[0] // 2
    dx = pk[1] - c.shape[1] // 2
    return shift(gt, [dy, dx], mode='wrap'), dy, dx

# ══════════════════════════════════════════════════════════════════
# 偵測
# ══════════════════════════════════════════════════════════════════
def detect_features(scan, shape_type, pct_lo, pct_hi, mn=5, mx=65536):
    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        thr  = np.percentile(scan, pct_lo)
        mask = scan < thr
    else:
        thr  = np.percentile(scan, pct_hi)
        mask = scan > thr
    lbl, n = label(mask)
    sz     = [(lbl == i).sum() for i in range(1, n + 1)]
    ids    = [i + 1 for i, s in enumerate(sz) if mn <= s <= mx]
    return (
        [(int(round(y)), int(round(x))) for y, x in center_of_mass(mask, lbl, ids)],
        n, len(ids),
    )

def compute_tip(avg, gt_p, shape_type, half, geom, px_nm, manual_offset=0.0):
    """從已配準(recenter)的 avg patch 計算探針形狀。

    不做任何深度縮放前處理，直接對 avg 做形態學腐蝕重建。
    manual_offset (nm) 純粹用於 cross-section 的視覺顯示（曲線上下移動），
    不影響 tip 計算。使用者可用滑桿觀察 AFM 曲線底部與 GT 底線的距離，
    判斷探針進入率與量測限制。

    回傳 (tip_sym, avg_display, rim, measured_depth)
      avg_display   : avg + manual_offset，供 cross-section 顯示用
      measured_depth: AFM 實際量到的特徵深度/高度 (nm，正值)
    """
    is_hole = shape_type in ('cylinder_hole', 'trapezoid_hole')

    # ── 表面/基底對齊到 GT 表面（不縮放，保留量測到的真實深度）─────
    # 孔洞：avg.max()≈0 對齊到 gt_p.max()=0，align_offset 通常≈0
    if is_hole:
        align_offset = gt_p.max() - avg.max()
    else:
        align_offset = gt_p.min() - avg.min()
    avg_aligned = avg + align_offset

    measured_depth = (avg_aligned.max() - avg_aligned.min())

    # ── 形態學腐蝕重建（Villarrubia，不加入任何深度補償）──────────
    tip_morph = reconstruct_morphological(avg_aligned, gt_p, shape_type)

    # ── 四向對稱化（旋轉對稱約束）──────────────────────────────
    tip_sym = (tip_morph
               + tip_morph[::-1, :]
               + tip_morph[:, ::-1]
               + tip_morph[::-1, ::-1]) / 4.0
    tip_sym -= tip_sym.max()

    # ── 徑向平均：強制完全旋轉對稱，消除 XY 殘差 ────────────────
    tip_sym = radial_average(tip_sym, half)

    # ── Rim 分析（不依賴孔底，低進入率仍有效）─────────────────
    rim = analyze_rim(avg_aligned, geom, shape_type, half, px_nm)

    # manual_offset 只用於視覺：把 cross-section 藍線上下移動供目視比對
    avg_display = avg_aligned + manual_offset

    return tip_sym, avg_display, rim, measured_depth


def reconstruct(scan, geom, shape_type, meta, ap, manual_offset=0.0):
    half   = ap['half']
    max_sz = ap['max_sz']
    # [FIX-3] 改用 is not None 避免 pct=0 被誤判為 falsy
    pct_lo = ap['pct_lo'] if ap['pct_lo'] is not None else 8
    pct_hi = ap['pct_hi'] if ap['pct_hi'] is not None else 92

    gt    = make_gt(meta['n_px'], meta['scan_nm'], geom, shape_type)
    gt_al, dy, dx = align_gt(scan, gt, shape_type)

    centers, n_raw, n_valid = detect_features(
        scan, shape_type, pct_lo, pct_hi, mn=5, mx=max_sz)

    padded    = np.pad(scan, half, mode='reflect')
    n         = scan.shape[0]
    patches   = []
    edge_cnt  = 0
    for cy, cx in centers:
        search = 8
        y0 = max(0, cy - search); y1 = min(n, cy + search)
        x0 = max(0, cx - search); x1 = min(n, cx + search)
        region = scan[y0:y1, x0:x1]
        if region.size > 0:
            if shape_type in ('cylinder_hole', 'trapezoid_hole'):
                loc = np.unravel_index(region.argmin(), region.shape)
            else:
                loc = np.unravel_index(region.argmax(), region.shape)
            cy, cx = y0 + loc[0], x0 + loc[1]
        cy_p, cx_p = cy + half, cx + half
        p = padded[cy_p - half:cy_p + half, cx_p - half:cx_p + half]
        if p.shape == (2 * half, 2 * half):
            patches.append(p)
            if min(cy, cx, n - 1 - cy, n - 1 - cx) < half:
                edge_cnt += 1

    edge_note = f' (含{edge_cnt}個邊緣特徵，用padding補全)' if edge_cnt else ''
    info = (
        f'偵測: {n_raw}個候選 → 過濾後{n_valid}個 → patch收集{len(patches)}個{edge_note}\n'
        f'  GT對齊: dy={dy}px dx={dx}px | half={half}px | max_sz={max_sz}px²'
    )

    if not patches:
        return None, None, None, 0, gt_al, info

    avg = np.mean(patches, axis=0)
    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        avg -= avg.max()   # 表面=0，孔洞往下
    else:
        avg -= avg.min()   # 基底=0，柱體往上

    gt_p = make_gt_patch(geom, shape_type, half, meta['px_nm'],
                          blur_sigma=0)   # 腐蝕法不需模糊，保持精確幾何

    # ── 2D 配準：將 avg 孔洞中心對齊到 GT 中心 ──────────────────
    avg, reg_dy, reg_dx = recenter_avg(avg, gt_p, shape_type)
    info += f'\n  2D配準偏移: dy={reg_dy}px dx={reg_dx}px'

    # ── 探針重建（不做深度縮放前處理，保留量測到的真實深度）──────
    tip_sym, avg_aligned, rim, meas_depth = compute_tip(
        avg, gt_p, shape_type, half, geom, meta['px_nm'], manual_offset)
    true_depth = abs(geom.get('height', 0.0))
    info += f'\n  量測特徵深度: {meas_depth:.1f}nm (GT真實: {true_depth:.1f}nm)'

    entry_pct = rim['entry_rate'] * 100
    info += (f'\n  進入率: {entry_pct:.1f}%  '
             f'Rim寬: {rim["rim_width_nm"]:.1f}nm  '
             f'Rim錐角: {rim["cone_angle_deg"]:.1f}°')
    if entry_pct < 50:
        info += f'\n  ⚠ 進入率<50%，孔底資訊不可靠，rim分析更準確'

    # 回傳 avg(配準後、未做底部對齊) 供 GUI 手動調整時快速重算（免重新偵測）
    return tip_sym, avg_aligned, gt_p, len(patches), gt_al, info, rim, avg

# ══════════════════════════════════════════════════════════════════
# 2D 影像配準：把 avg patch 的孔洞中心對齊到 GT 中心
# ══════════════════════════════════════════════════════════════════
def recenter_avg(avg, gt_p, shape_type):
    """
    用 2D 互相關找出 avg 相對於 GT 的偏移，
    然後平移 avg，使孔洞/特徵中心與 GT 中心對齊。
    回傳 (avg_centered, dy, dx)
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import shift as ndshift

    # 孔洞型：都是負值，取絕對值後做相關（讓孔洞成為正峰）
    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        a = -avg.copy()
        g = -gt_p.copy()
    else:
        a = avg.copy()
        g = gt_p.copy()

    # 確保非負
    a -= a.min(); g -= g.min()

    # fftconvolve(a, g[::-1,::-1]) 等效 correlate2d(a, g)
    corr = fftconvolve(a, g[::-1, ::-1], mode='same')
    pk   = np.unravel_index(corr.argmax(), corr.shape)
    dy   = pk[0] - corr.shape[0] // 2
    dx   = pk[1] - corr.shape[1] // 2

    # 把 avg 往反方向移，使特徵中心貼齊 GT 中心
    avg_centered = ndshift(avg, [-dy, -dx], mode='nearest')
    return avg_centered, dy, dx


# ══════════════════════════════════════════════════════════════════
# 徑向平均：強制旋轉對稱（消除 XY 不對稱殘差）
# ══════════════════════════════════════════════════════════════════
def radial_average(tip, half):
    """
    以中心為原點，把 tip 2D 陣列轉換為徑向平均結果。
    每個像素的值 = 同半徑圓環上所有像素的平均值。
    物理假設：探針尖端是旋轉對稱的。
    """
    size = 2 * half
    yi, xi = np.indices((size, size))
    r_px = np.sqrt((xi - half)**2 + (yi - half)**2)
    r_int = np.round(r_px).astype(int)

    tip_radial = np.zeros_like(tip)
    for r in range(0, half + 1):
        mask = r_int == r
        if mask.sum() > 0:
            tip_radial[mask] = tip[mask].mean()

    # 超出 half 半徑範圍的角落像素，用最外圈值填補
    outer_mask = r_int > half
    if outer_mask.sum() > 0:
        edge_val = tip[r_int == half].mean() if (r_int == half).sum() > 0 else 0
        tip_radial[outer_mask] = edge_val

    tip_radial -= tip_radial.max()  # 尖端歸零
    return tip_radial


# ══════════════════════════════════════════════════════════════════
# 形態學腐蝕重建（Villarrubia 算法）
# ══════════════════════════════════════════════════════════════════
def reconstruct_morphological(avg_patch, gt_p, shape_type):
    """
    正確的盲探針重建：形態學腐蝕
    原理：AFM掃描 = 真實表面 ⊕ 探針  → 探針 = 掃描 ⊖ 表面
    對孔洞：先反轉（孔洞變峰），再腐蝕，再反轉回來
    """
    from scipy.ndimage import grey_erosion

    if shape_type in ('cylinder_hole', 'trapezoid_hole'):
        # 反轉：孔洞（負值）變峰（正值）
        avg_inv = -(avg_patch.copy())
        avg_inv -= avg_inv.min()           # 確保全非負
        gt_inv  = -(gt_p.copy())
        gt_inv  -= gt_inv.min()

        # 形態學腐蝕（GT patch 當 structure）
        # grey_erosion：output[i,j] = min over structure { input[i+dy,j+dx] - structure[dy,dx] }
        try:
            tip_inv = grey_erosion(avg_inv, structure=gt_inv)
            tip = -tip_inv
        except MemoryError:
            # structure 太大時退回簡單版本
            tip = gt_p - avg_patch
    else:
        avg_pos = avg_patch.copy(); avg_pos -= avg_pos.min()
        gt_pos  = gt_p.copy();    gt_pos  -= gt_pos.min()
        try:
            tip = grey_erosion(avg_pos, structure=gt_pos)
        except MemoryError:
            tip = avg_patch - gt_p

    tip -= tip.max()
    return tip


def analyze_rim(avg_patch, geom, shape_type, half, px_nm):
    """
    從孔洞邊緣過渡帶分析探針有效半徑與錐角。
    不依賴孔底資訊，在進入率低時仍有效。
    回傳：dict 包含 rim_width_nm, cone_angle_deg, entry_rate
    """
    from scipy.optimize import curve_fit

    # 建立平均 radial profile（以中心為圓心）
    size = avg_patch.shape[0]
    mid  = half
    yi, xi = np.indices((size, size))
    r_px = np.sqrt((xi - mid)**2 + (yi - mid)**2)
    r_nm = r_px * px_nm

    # 取同心環平均（bin 寬 = 1px）
    r_max_px = int(np.sqrt(2) * half)
    r_bins   = np.arange(0, r_max_px + 1)
    profile  = np.zeros(len(r_bins) - 1)
    r_centers= np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins)-1):
        ring = (r_px >= r_bins[i]) & (r_px < r_bins[i+1])
        if ring.sum() > 0:
            profile[i]   = avg_patch[ring].mean()
            r_centers[i] = (r_bins[i] + r_bins[i+1]) / 2 * px_nm

    # 找過渡帶：從孔底到表面的 20%-80% 位置
    p_min, p_max = profile.min(), profile.max()
    p_range = p_max - p_min
    if p_range < 1:
        return {'rim_width_nm': float('nan'), 'cone_angle_deg': float('nan'),
                'entry_rate': 0., 'r_centers': r_centers, 'profile': profile}

    p_norm  = (profile - p_min) / p_range
    idx_20  = np.argmax(p_norm > 0.2) if (p_norm > 0.2).any() else 0
    idx_80  = np.argmax(p_norm > 0.8) if (p_norm > 0.8).any() else len(p_norm)-1

    rim_width_nm = abs(r_centers[idx_80] - r_centers[idx_20])

    # 在過渡帶內線性擬合斜率 → 錐角
    idx_lo = min(idx_20, idx_80); idx_hi = max(idx_20, idx_80)
    if idx_hi > idx_lo + 1:
        r_rim = r_centers[idx_lo:idx_hi+1]
        p_rim = profile[idx_lo:idx_hi+1]
        try:
            slope = np.polyfit(r_rim, p_rim, 1)[0]  # nm height / nm radius
            cone_angle = float(np.degrees(np.arctan(abs(slope))))
        except Exception:
            cone_angle = float('nan')
    else:
        cone_angle = float('nan')

    entry_rate = abs(p_min) / geom['height'] if geom['height'] > 0 else 0.

    return {
        'rim_width_nm'  : rim_width_nm,
        'cone_angle_deg': cone_angle,
        'entry_rate'    : entry_rate,
        'r_centers'     : r_centers,
        'profile'       : profile,
    }


# ══════════════════════════════════════════════════════════════════
# 錐形模型（廠商規格約束）
# ══════════════════════════════════════════════════════════════════
def make_cone_tip(half, px_nm, R_nm, theta_deg):
    """
    球形尖端 + 錐形側壁的理論探針形狀。
    R_nm     : 尖端球形半徑（nm）
    theta_deg: 半錐角（°），從探針軸量起
    回傳：2D array，中心=0，往外為負值（nm）
    """
    size = 2 * half
    yi, xi = np.indices((size, size))
    r_nm   = np.sqrt((xi - half) ** 2 + (yi - half) ** 2) * px_nm

    theta   = np.radians(theta_deg)
    r_trans = R_nm * np.sin(theta)
    tip     = np.zeros((size, size))

    sphere_mask = r_nm <= r_trans
    tip[sphere_mask] = -(R_nm - np.sqrt(
        np.maximum(R_nm ** 2 - r_nm[sphere_mask] ** 2, 0)))

    # [FIX-6] 改用 np.maximum 確保語義一致（雖然此處 r_trans 是 scalar 仍安全）
    z_trans = -(R_nm - np.sqrt(np.maximum(R_nm ** 2 - r_trans ** 2, 0.0)))
    cone_mask = r_nm > r_trans
    tip[cone_mask] = z_trans - (r_nm[cone_mask] - r_trans) / np.tan(theta)

    tip = np.clip(tip, -half * px_nm * 2, 0)
    tip -= tip.max()
    return tip

# ══════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.FT = get_fonts()
        self.title('AFM Blind Tip Reconstruction  v4.1')
        self.configure(bg=C['bg'])
        self.geometry('1420x920')
        self.minsize(1150, 740)

        self.meta       = None
        self.scan       = None
        self.tip        = None          # [FIX-7] _preview() 會重設此值
        self.tip_half   = None
        self.geom       = None
        self.shape_type = None

        self.afm_path  = tk.StringVar(value='尚未選擇')
        self.ch_var    = tk.StringVar()
        self.shape_var = tk.StringVar(value='trapezoid_hole')
        self.d_bot     = tk.StringVar(value='183.5')
        self.d_top     = tk.StringVar(value='228.8')
        self.height    = tk.StringVar(value='125.8')
        self.pitch     = tk.StringVar(value='1000')
        self.half_v    = tk.StringVar(value='auto')
        self.max_sz_v  = tk.StringVar(value='auto')
        self.pct_v     = tk.StringVar(value='auto')
        self.tip_R_v   = tk.StringVar(value='2')
        self.tip_angle_v = tk.StringVar(value='25')

        # 手動調整 AFM 曲線深度（探底）用
        self.offset_var      = tk.DoubleVar(value=0.0)
        self.recon_ctx       = None    # 儲存重建中間結果，供手動調整快速重算
        self._offset_job     = None    # 防抖動 after job id
        self._suppress_offset = False  # 程式設定 offset 時抑制重算

        for v in (self.d_bot, self.d_top, self.height):
            v.trace_add('write', lambda *_: self.after(80, self._update_info))

        self._build()
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    # ── 佈局 ──────────────────────────────────────────────────────
    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._sidebar()
        self._main_area()

    def _sidebar(self):
        # [FIX-9] 用 Canvas + Scrollbar 包住 sidebar，支援捲動
        # 外層容器（固定寬度）
        outer = tk.Frame(self, bg=C['panel'], width=340)
        outer.grid(row=0, column=0, sticky='nsew')
        outer.grid_propagate(False)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        # 捲動用 Canvas
        self._sb_canvas = tk.Canvas(outer, bg=C['panel'], highlightthickness=0,
                                    width=320)
        self._sb_canvas.grid(row=0, column=0, sticky='nsew')

        sb_scroll = ttk.Scrollbar(outer, orient='vertical',
                                  command=self._sb_canvas.yview)
        sb_scroll.grid(row=0, column=1, sticky='ns')
        self._sb_canvas.configure(yscrollcommand=sb_scroll.set)

        # 真正的 sidebar Frame 放在 Canvas 裡
        sb = tk.Frame(self._sb_canvas, bg=C['panel'])
        self._sb_win = self._sb_canvas.create_window(
            (0, 0), window=sb, anchor='nw', width=320)

        # 內容尺寸變化時更新 scrollregion
        def _on_frame_configure(event):
            self._sb_canvas.configure(
                scrollregion=self._sb_canvas.bbox('all'))
        sb.bind('<Configure>', _on_frame_configure)

        # Canvas 寬度調整時同步 window 寬度
        def _on_canvas_configure(event):
            self._sb_canvas.itemconfig(self._sb_win, width=event.width)
        self._sb_canvas.bind('<Configure>', _on_canvas_configure)

        # 滑鼠滾輪（Windows / macOS / Linux）
        def _on_mousewheel(event):
            if event.delta:                          # Windows / macOS
                self._sb_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            elif event.num == 4:                     # Linux scroll up
                self._sb_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:                     # Linux scroll down
                self._sb_canvas.yview_scroll(1, 'units')
        self._sb_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        self._sb_canvas.bind_all('<Button-4>',   _on_mousewheel)
        self._sb_canvas.bind_all('<Button-5>',   _on_mousewheel)

        sb.columnconfigure(0, weight=1)

        # ── 以下是原 sidebar 內容，父容器改為 sb ─────────────────
        tk.Label(sb, text='AFM Tip', font=self.FT['title'],
                 bg=C['panel'], fg=C['accent']).pack(pady=(16, 0))
        tk.Label(sb, text='Reconstruction  v4.2', font=self.FT['normal'],
                 bg=C['panel'], fg=C['muted']).pack(pady=(0, 10))
        ttk.Separator(sb).pack(fill='x', padx=14)

        # ① AFM
        self._sec(sb, '① 選擇 AFM 檔案')
        pf = tk.Frame(sb, bg=C['panel'])
        pf.pack(fill='x', padx=14, pady=(0, 6))
        pf.columnconfigure(0, weight=1)
        tk.Label(pf, textvariable=self.afm_path, bg=C['input_bg'], fg=C['text'],
                 font=self.FT['mono'], wraplength=210, anchor='w', padx=6, pady=5
                 ).grid(row=0, column=0, sticky='ew', padx=(0, 6))
        self._btn(pf, '瀏覽', self.browse, small=True).grid(row=0, column=1)
        tk.Label(sb, text='高度通道', bg=C['panel'], fg=C['muted'],
                 font=self.FT['small']).pack(anchor='w', padx=14)
        self.ch_cb = ttk.Combobox(sb, textvariable=self.ch_var, state='readonly',
                                  font=self.FT['small'])
        self.ch_cb.pack(fill='x', padx=14, pady=(2, 10))
        self.ch_cb.bind('<<ComboboxSelected>>', lambda _: self._preview())
        ttk.Separator(sb).pack(fill='x', padx=14)

        # ② 形狀
        self._sec(sb, '② 特徵形狀')
        sf = tk.Frame(sb, bg=C['panel'])
        sf.pack(fill='x', padx=14, pady=(0, 6))
        for txt, val in [('梯形孔 Trapezoid Hole',   'trapezoid_hole'),
                         ('梯形柱 Trapezoid Pillar', 'trapezoid_pillar'),
                         ('圓柱孔 Cylinder Hole',    'cylinder_hole')]:
            tk.Radiobutton(sf, text=txt, variable=self.shape_var, value=val,
                           bg=C['panel'], fg=C['text'], selectcolor=C['input_bg'],
                           activebackground=C['panel'], font=self.FT['small'],
                           command=self._on_shape).pack(anchor='w', pady=2)
        ttk.Separator(sb).pack(fill='x', padx=14)

        # ③ 幾何
        self._sec(sb, '③ SEM 幾何參數')
        self.geom_frame = tk.Frame(sb, bg=C['panel'])
        self.geom_frame.pack(fill='x')
        self._rebuild_geom()
        ttk.Separator(sb).pack(fill='x', padx=14)

        # ④ 進階
        self._sec(sb, '④ 進階設定（auto = 自動計算）')
        self._row(sb, 'Patch 半徑 (px)',    self.half_v)
        self._row(sb, '最大特徵面積 (px²)', self.max_sz_v)
        self._row(sb, '偵測閾值 (%)',        self.pct_v)
        self.auto_info = tk.Label(sb, text='', bg=C['panel'], fg=C['warn'],
                                  font=self.FT['mono'], wraplength=295, justify='left')
        self.auto_info.pack(anchor='w', padx=14, pady=(4, 0))
        ttk.Separator(sb).pack(fill='x', padx=14)

        # ⑤ 廠商規格
        self._sec(sb, '⑤ 探針廠商規格（選填）')
        tk.Label(sb, text='  填入後自動疊加錐形模型做比較',
                 bg=C['panel'], fg=C['muted'], font=self.FT['tiny']
                 ).pack(anchor='w', padx=14)
        self._row(sb, '尖端半徑 R  (nm)', self.tip_R_v)
        self._row(sb, '半錐角 θ    (°)',  self.tip_angle_v)
        self.tip_info_lbl = tk.Label(sb, text='', bg=C['panel'], fg=C['success'],
                                     font=self.FT['mono'], wraplength=295, justify='left')
        self.tip_info_lbl.pack(anchor='w', padx=14, pady=(2, 0))
        for v in (self.tip_R_v, self.tip_angle_v):
            v.trace_add('write', lambda *_: self.after(80, self._update_tip_spec))
        ttk.Separator(sb).pack(fill='x', padx=14)

        # ⑥ 手動平移 AFM 曲線
        self._sec(sb, '⑥ 手動平移 AFM 曲線')
        tk.Label(sb, text=('  拖曳滑桿或點按「▼ 下 / ▲ 上」移動藍色\n'
                           '  AFM 曲線，觀察底部與紅色 GT 底線距離\n'
                           '  即可判斷探針實際進入率'),
                 bg=C['panel'], fg=C['muted'], font=self.FT['tiny'],
                 justify='left').pack(anchor='w', padx=14)

        of = tk.Frame(sb, bg=C['panel'])
        of.pack(fill='x', padx=14, pady=(4, 2))
        of.columnconfigure(1, weight=1)
        self._btn(of, '▼ 下', lambda: self._nudge_offset(-1), small=True
                  ).grid(row=0, column=0)
        self.offset_scale = tk.Scale(
            of, variable=self.offset_var, from_=-200, to=200, resolution=1,
            orient='horizontal', bg=C['panel'], fg=C['text'],
            troughcolor=C['input_bg'], highlightthickness=0, showvalue=False,
            sliderrelief='flat', bd=0,
            command=lambda _=None: self._on_offset_change())
        self.offset_scale.grid(row=0, column=1, sticky='ew', padx=4)
        self._btn(of, '▲ 上', lambda: self._nudge_offset(+1), small=True
                  ).grid(row=0, column=2)

        orow = tk.Frame(sb, bg=C['panel'])
        orow.pack(fill='x', padx=14, pady=(0, 4))
        orow.columnconfigure(0, weight=1)
        self.offset_lbl = tk.Label(orow, text='±0 nm', bg=C['panel'],
                                   fg=C['accent'], font=self.FT['small'],
                                   anchor='w')
        self.offset_lbl.grid(row=0, column=0, sticky='w')
        self._btn(orow, '歸零', self._reset_offset, small=True
                  ).grid(row=0, column=1)

        tk.Frame(sb, bg=C['panel'], height=8).pack()
        self._btn(sb, '▶  執行重建', self.run, big=True
                  ).pack(fill='x', padx=14, pady=3)
        self._btn(sb, '💾  儲存 tip.mat', self.save, color=C['success']
                  ).pack(fill='x', padx=14, pady=3)
        self._btn(sb, '🟠  儲存 Cone 為 tip.mat', self.save_cone, color=C['warn']
                  ).pack(fill='x', padx=14, pady=3)
        self._btn(sb, '🖼  儲存圖片', self.save_figure, color=C['accent2']
                  ).pack(fill='x', padx=14, pady=3)

        ttk.Separator(sb).pack(fill='x', padx=14)
        tk.Label(sb, text='LOG', bg=C['panel'], fg=C['muted'],
                 font=self.FT['tiny']).pack(anchor='w', padx=14, pady=(6, 2))
        lf = tk.Frame(sb, bg=C['input_bg'], height=160)
        lf.pack(fill='x', padx=14, pady=(0, 14))
        lf.pack_propagate(False)
        sc = ttk.Scrollbar(lf, orient='vertical')
        self.log_box = tk.Text(lf, bg=C['input_bg'], fg='#94a3b8',
                               font=self.FT['mono'], relief='flat',
                               state='disabled', wrap='word',
                               yscrollcommand=sc.set, height=8)
        sc.configure(command=self.log_box.yview)
        sc.pack(side='right', fill='y')
        self.log_box.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        # 底部留白，確保捲動到底時最後一個元件不被截斷
        tk.Frame(sb, bg=C['panel'], height=12).pack()

    def _rebuild_geom(self):
        for w in self.geom_frame.winfo_children():
            w.destroy()
        s = self.shape_var.get()
        if s in ('trapezoid_hole', 'trapezoid_pillar'):
            self.angle_lbl = tk.Label(self.geom_frame, text='', bg=C['panel'],
                                      fg=C['warn'], font=self.FT['small'])
            self.angle_lbl.pack(anchor='w', padx=14)
            if s == 'trapezoid_hole':
                fields = [('開口寬 d_top (nm)', self.d_top),
                          ('底部寬 d_bot (nm)', self.d_bot),
                          ('深度 depth  (nm)',  self.height),
                          ('週期 pitch  (nm)',  self.pitch)]
            else:
                fields = [('底部寬 d_bot (nm)', self.d_bot),
                          ('頂部寬 d_top (nm)', self.d_top),
                          ('高度 height  (nm)', self.height),
                          ('週期 pitch   (nm)', self.pitch)]
        else:
            self.angle_lbl = None
            fields = [('孔洞直徑 d  (nm)', self.d_bot),
                      ('孔洞深度 z  (nm)', self.height),
                      ('週期 pitch  (nm)', self.pitch)]
        for lbl, var in fields:
            self._row(self.geom_frame, lbl, var)
        self._update_info()

    def _on_shape(self):
        self._rebuild_geom()
        # shape_type 改變時需重新 flatten（基線方向不同）
        if self.meta and self.scan is not None:
            raw = read_channel(self.meta['filepath'], self.meta, self.ch_cb.current())
            self.scan = flatten_rows(raw, self.shape_var.get())
            self._draw_preview()

    def _update_info(self):
        if hasattr(self, 'angle_lbl') and self.angle_lbl:
            try:
                bot = float(self.d_bot.get() or 0)
                top = float(self.d_top.get() or 0)
                h   = float(self.height.get() or 0)
                if h > 0 and abs(top - bot) > 0:
                    taper = np.degrees(np.arctan(abs(top - bot) / 2 / h))
                    if self.shape_var.get() == 'trapezoid_hole':
                        kind = '梯形孔(開口寬底部窄)'
                    else:
                        kind = '倒梯形(底窄頂寬)' if top > bot else '正梯形(底寬頂窄)'
                    self.angle_lbl.config(text=f'  {kind}  taper: {taper:.1f}°')
            except Exception:
                pass
        if self.meta:
            try:
                geom = self._get_geom()
                ap   = auto_params(geom, self.shape_var.get(), self.meta)
                info = (f'auto: half={ap["half"]}px  max_sz={ap["max_sz"]}px²\n'
                        f'      pct={ap["pct_hi"] or ap["pct_lo"]}%  '
                        f'r_feat={ap["r_max_px"]:.1f}px')
                if ap['warnings']:
                    info += '\n' + '\n'.join(ap['warnings'])
                self.auto_info.config(text=info)
            except Exception:
                pass

    def _update_tip_spec(self):
        try:
            R     = float(self.tip_R_v.get() or 0)
            theta = float(self.tip_angle_v.get() or 0)
            if R > 0 and theta > 0:
                r_t = R * np.sin(np.radians(theta))
                d_t = R * (1 - np.cos(np.radians(theta)))
                self.tip_info_lbl.config(text=(
                    f'  尖端球形區：r < {r_t:.1f}nm\n'
                    f'  切入深度:    {d_t:.1f}nm\n'
                    f'  側壁斜率:    {np.tan(np.radians(theta)):.3f} nm/nm'
                ))
            elif R > 0:
                self.tip_info_lbl.config(text='  請也輸入半錐角 θ')
            else:
                self.tip_info_lbl.config(text='')
        except Exception:
            pass

    # ── 手動調整 AFM 曲線深度（探底）────────────────────────────────
    def _nudge_offset(self, direction):
        """以一個 pixel(若已知)或 1nm 為步距上下微調 AFM 曲線。"""
        step = 1.0
        if self.recon_ctx:
            step = max(1.0, round(self.recon_ctx['px']))
        try:
            cur = float(self.offset_var.get())
        except (ValueError, tk.TclError):
            cur = 0.0
        self.offset_var.set(round(cur + direction * step, 1))
        # tk.Scale.set() 不會觸發 command callback，必須明確呼叫
        self._on_offset_change()

    def _reset_offset(self):
        self.offset_var.set(0.0)
        self._on_offset_change()

    def _on_offset_change(self, *_):
        """偏移改變：更新標籤並防抖動排程重算（避免拖曳時卡頓）。"""
        if self._suppress_offset:
            return
        try:
            v = float(self.offset_var.get())
        except (ValueError, tk.TclError):
            v = 0.0
        if hasattr(self, 'offset_lbl'):
            tag = ('下移' if v < 0 else '上移' if v > 0 else '±')
            self.offset_lbl.config(text=f'{tag}{abs(v):.0f} nm')
        if self.recon_ctx is None:
            return
        if self._offset_job is not None:
            self.after_cancel(self._offset_job)
        self._offset_job = self.after(180, self._recompute_tip)

    def _sync_offset_range(self):
        """重建完成後，依特徵高度設定滑桿範圍並把偏移歸零。"""
        self._suppress_offset = True
        try:
            h = float(self.height.get() or 0)
        except (ValueError, tk.TclError):
            h = 0.0
        if h > 0 and hasattr(self, 'offset_scale'):
            rng = max(50, int(round(h * 1.5)))
            self.offset_scale.config(from_=-rng, to=rng)
        self.offset_var.set(0.0)
        if hasattr(self, 'offset_lbl'):
            self.offset_lbl.config(text='±0 nm')
        self._suppress_offset = False

    def _recompute_tip(self):
        """手動偏移改變後，只重算探針（不重新偵測特徵），即時更新顯示。"""
        self._offset_job = None
        ctx = self.recon_ctx
        if not ctx or ctx.get('avg') is None:
            return
        try:
            offset = float(self.offset_var.get())
        except (ValueError, tk.TclError):
            return

        def worker():
            try:
                tip, avg_disp, rim, meas_d = compute_tip(
                    ctx['avg'], ctx['gt_p'], ctx['shape'], ctx['half'],
                    ctx['geom'], ctx['px'], manual_offset=offset)
                half = ctx['half']; px = ctx['px']
                tip1d  = tip[half, :]
                beyond = np.where(tip1d < tip.min() * 0.1)[0]
                r_est  = ((beyond[-1] - beyond[0]) * px / 2
                          if len(beyond) > 1 else float('nan'))
                self.tip      = tip
                self.tip_half = half
                self.after(0, lambda: self._draw_result(
                    tip, avg_disp, ctx['gt_p'], ctx['gt_al'],
                    ctx['geom'], ctx['shape'], ctx['n'], r_est, half, px, rim))
                tag = ('下移' if offset < 0 else '上移' if offset > 0 else '原位')
                msg = (f'  {tag} {abs(offset):.0f}nm  量測深度:{meas_d:.1f}nm  '
                       f'進入率:{rim["entry_rate"]*100:.1f}%')
                if not np.isnan(r_est):
                    msg += f'  tip半徑:{r_est:.1f}nm'
                self._log_safe(msg)
            except Exception as e:
                self._log_safe(f'✗ 重算失敗: {e}')

        threading.Thread(target=worker, daemon=True).start()

    def _main_area(self):
        m = tk.Frame(self, bg=C['bg'])
        m.grid(row=0, column=1, sticky='nsew', padx=14, pady=14)
        m.columnconfigure(0, weight=1)
        m.rowconfigure(1, weight=1)

        sb = tk.Frame(m, bg=C['panel'], height=44)
        sb.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        self.stats = {}
        for i, (k, v) in enumerate([('AFM', '—'), ('掃描範圍', '—'),
                                     ('px_nm', '—'), ('Ra', '—'), ('特徵數', '—')]):
            f = tk.Frame(sb, bg=C['panel'])
            f.grid(row=0, column=i, padx=16, pady=8)
            tk.Label(f, text=k, bg=C['panel'], fg=C['muted'],
                     font=self.FT['tiny']).pack()
            lbl = tk.Label(f, text=v, bg=C['panel'], fg=C['text'],
                           font=self.FT['bold'])
            lbl.pack()
            self.stats[k] = lbl

        self.fig    = Figure(figsize=(10, 6.2), facecolor=C['bg'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=m)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self._placeholder()

    def _sec(self, p, t):
        tk.Label(p, text=t, bg=C['panel'], fg=C['accent'],
                 font=self.FT['bold']).pack(anchor='w', padx=14, pady=(9, 3))

    def _row(self, p, lbl_text, var):
        f = tk.Frame(p, bg=C['panel'])
        f.pack(fill='x', padx=14, pady=2)
        f.columnconfigure(1, weight=1)
        tk.Label(f, text=lbl_text, bg=C['panel'], fg=C['muted'],
                 font=self.FT['small'], width=19, anchor='w').grid(row=0, column=0)
        make_float_entry(f, var, font=self.FT['normal']
                         ).grid(row=0, column=1, sticky='ew', ipady=4, padx=(6, 0))

    def _btn(self, p, t, cmd, big=False, small=False, color=None):
        c = color or C['accent']
        return tk.Button(p, text=t, command=cmd, bg=c, fg='white',
                         activebackground=C['hover'], activeforeground='white',
                         relief='flat', cursor='hand2',
                         font=(self.FT['title'] if big else
                               (self.FT['small'] if small else self.FT['normal'])),
                         pady=8 if big else 4)

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(C['panel'])
        ax.text(0.5, 0.5, '選擇 AFM 檔案後開始分析',
                ha='center', va='center', color=C['muted'],
                fontsize=14, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C['border'])
        self.canvas.draw()

    # [FIX-2] 只能在主執行緒操作 Tkinter widget；worker 透過 self.after() 排程
    def log(self, msg):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', msg + '\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _log_safe(self, msg):
        """thread-safe 版本：從 worker thread 呼叫此方法"""
        self.after(0, lambda m=msg: self.log(m))

    def _get_geom(self):
        s = self.shape_var.get()
        return {
            'd_bot':   float(self.d_bot.get()),
            'd_top':   float(self.d_top.get()) if s in ('trapezoid_pillar', 'trapezoid_hole')
                       else float(self.d_bot.get()),
            'height':  float(self.height.get()),
            'pitch':   float(self.pitch.get()),
        }

    def browse(self):
        p = filedialog.askopenfilename(
            title='選擇 Nanoscope 檔案',
            filetypes=[('Nanoscope', '*.000 *.001 *.002 *.003 *.004 *.005'),
                       ('All', '*.*')])
        if not p: return
        try:
            self.meta = parse_nanoscope(p)
            self.afm_path.set(os.path.basename(p))
            names = [f'[{i}] {ch["name"]}' for i, ch in enumerate(self.meta['channels'])]
            self.ch_cb['values'] = names
            auto = auto_height_ch(self.meta)
            self.ch_cb.current(auto)
            self.ch_var.set(names[auto])
            self._preview()
            self._update_info()
            self.log(f'✓ {os.path.basename(p)} | {len(self.meta["channels"])}通道 | '
                     f'{self.meta["scan_nm"]:.1f}nm | {self.meta["n_px"]}px')
        except Exception as e:
            messagebox.showerror('讀取失敗', str(e))
            self.log(f'✗ {e}')

    # [FIX-7] 切換通道時清除舊 tip，避免誤存
    def _preview(self):
        if not self.meta: return
        self.tip        = None      # 清除舊重建結果
        self.tip_half   = None
        self.stats['特徵數'].config(text='—')

        # [FIX-11] 把目前選擇的 shape_type 傳給 flatten_rows，讓基線方向正確
        raw       = read_channel(self.meta['filepath'], self.meta, self.ch_cb.current())
        self.scan = flatten_rows(raw, self.shape_var.get())
        m = self.meta; s = self.scan
        self.stats['AFM'].config(text=os.path.basename(m['filepath']))
        self.stats['掃描範圍'].config(text=f'{m["scan_nm"]:.1f}nm')
        self.stats['px_nm'].config(text=f'{m["px_nm"]:.2f}nm/px')
        self.stats['Ra'].config(text=f'{np.mean(np.abs(s - s.mean())):.2f}nm')
        self._draw_preview()

    def _draw_preview(self):
        self.fig.clear()
        self.fig.patch.set_facecolor(C['bg'])
        gs = self.fig.add_gridspec(1, 2, wspace=0.3,
                                   left=0.06, right=0.97, top=0.9, bottom=0.1)
        s = self.scan; m = self.meta
        ext = [0, m['scan_nm'], m['scan_nm'], 0]

        def da(ax, t):
            ax.set_facecolor(C['panel'])
            ax.set_title(t, color=C['text'], fontsize=10)
            ax.tick_params(colors=C['muted'], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(C['border'])

        ax1 = self.fig.add_subplot(gs[0])
        da(ax1, 'AFM Scan (Height Sensor)')
        im = ax1.imshow(s, cmap='afmhot', origin='upper', extent=ext,
                        vmin=np.percentile(s, 2), vmax=np.percentile(s, 98))
        cb = self.fig.colorbar(im, ax=ax1, fraction=0.046)
        cb.ax.tick_params(colors='#666', labelsize=6)
        ax1.set_xlabel('nm', color=C['muted'], fontsize=8)
        ax1.set_ylabel('nm', color=C['muted'], fontsize=8)

        ax2 = self.fig.add_subplot(gs[1])
        da(ax2, 'Line Profiles')
        x = np.linspace(0, m['scan_nm'], s.shape[1])
        for row, col in zip([s.shape[0] // 4, s.shape[0] // 2, 3 * s.shape[0] // 4],
                            ['#4f9cf9', '#f59e0b', '#ef4444']):
            ax2.plot(x, s[row, :], color=col, lw=1.2, alpha=0.8, label=f'row {row}')
        ax2.set_xlabel('nm', color=C['muted'], fontsize=8)
        ax2.set_ylabel('nm', color=C['muted'], fontsize=8)
        ax2.grid(color=C['border'], alpha=0.5)
        leg = ax2.legend(fontsize=7, facecolor=C['panel'], edgecolor=C['border'])
        [t.set_color(C['text']) for t in leg.get_texts()]
        self.canvas.draw()

    def run(self):
        if self.scan is None:
            messagebox.showwarning('提示', '請先選擇 AFM 檔案')
            return
        try:
            geom = self._get_geom()
            s    = self.shape_var.get()
            ap   = auto_params(geom, s, self.meta)
            if self.half_v.get() not in ('', 'auto'):
                ap['half'] = int(float(self.half_v.get()))
            if self.max_sz_v.get() not in ('', 'auto'):
                ap['max_sz'] = int(float(self.max_sz_v.get()))
            if self.pct_v.get() not in ('', 'auto'):
                pct = float(self.pct_v.get())
                if s in ('cylinder_hole', 'trapezoid_hole'):
                    ap['pct_lo'] = pct
                else:
                    ap['pct_hi'] = pct
        except ValueError:
            messagebox.showerror('錯誤', '請確認所有參數為有效數字')
            return

        self.log(f'執行重建 shape={s}')
        self.log(f'  d_bot={geom["d_bot"]} d_top={geom["d_top"]} '
                 f'h={geom["height"]} p={geom["pitch"]}')
        self.log(f'  half={ap["half"]}px  max_sz={ap["max_sz"]}px²  '
                 f'pct={ap["pct_hi"] or ap["pct_lo"]}%')
        for w in ap['warnings']:
            self.log(f'  {w}')

        def worker():
            try:
                _result = reconstruct(self.scan, geom, s, self.meta, ap)
                tip, avg, gt_p, n, gt_al, info = _result[:6]
                rim = _result[6] if len(_result) > 6 else None
                avg_rc = _result[7] if len(_result) > 7 else None
                # [FIX-2] 所有 self.log() 改用 thread-safe 版本
                self._log_safe(info)
                if tip is None:
                    self.after(0, lambda: messagebox.showerror(
                        '失敗',
                        '找不到有效特徵\n\n建議：\n'
                        '1. 確認形狀選擇正確\n'
                        '2. 調整偵測閾值\n'
                        '3. 縮小掃描範圍讓特徵更明顯'))
                    self._log_safe('✗ 找不到有效特徵')
                    return

                self.tip        = tip
                self.tip_half   = ap['half']
                self.geom       = geom
                self.shape_type = s
                px  = self.meta['px_nm']
                mid = ap['half']
                tip1d  = tip[mid, :]
                beyond = np.where(tip1d < tip.min() * 0.1)[0]
                r_est  = (beyond[-1] - beyond[0]) * px / 2 if len(beyond) > 1 else float('nan')
                feat_h = abs(avg.min() if s in ('cylinder_hole', 'trapezoid_hole')
                             else avg.max())
                self._log_safe(
                    f'✓ {n}個特徵平均 | 量測高度:{feat_h:.1f}nm (GT:{geom["height"]}nm)')
                if not np.isnan(r_est):
                    self._log_safe(f'  估算探針半徑:{r_est:.1f}nm')
                # 儲存中間結果，供「⑥ 手動調整」即時重算（免重新偵測）
                self.recon_ctx = {
                    'avg': avg_rc, 'gt_p': gt_p, 'gt_al': gt_al,
                    'n': n, 'half': ap['half'], 'px': px,
                    'geom': geom, 'shape': s,
                }
                self.after(0, self._sync_offset_range)
                self.after(0, lambda: self._draw_result(
                    tip, avg, gt_p, gt_al, geom, s, n, r_est, ap['half'], px, rim))
                self.after(0, lambda: self.stats['特徵數'].config(text=str(n)))
            except Exception as e:
                self._log_safe(f'✗ {e}')
                import traceback; traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    def _draw_result(self, tip, avg, gt_p, gt_al, geom, shape, n, r_est, half, px, rim=None):
        self.fig.clear()
        self.fig.patch.set_facecolor(C['bg'])
        gs = self.fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                                   left=0.06, right=0.97, top=0.92, bottom=0.07)
        m   = self.meta
        ext = [0, m['scan_nm'], m['scan_nm'], 0]
        s   = self.scan

        def da(ax, t):
            ax.set_facecolor(C['panel'])
            ax.set_title(t, color=C['text'], fontsize=9, pad=4)
            ax.tick_params(colors=C['muted'], labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(C['border'])

        def dcb(im, ax):
            c = self.fig.colorbar(im, ax=ax, fraction=0.046)
            c.ax.tick_params(colors='#666', labelsize=5)

        ax1 = self.fig.add_subplot(gs[0, 0]); da(ax1, 'AFM Scan')
        im  = ax1.imshow(s, cmap='afmhot', origin='upper', extent=ext,
                         vmin=np.percentile(s, 2), vmax=np.percentile(s, 98))
        dcb(im, ax1); ax1.set_xlabel('nm', color=C['muted'], fontsize=7)

        if shape == 'trapezoid_hole':
            lbl = f'開口={geom["d_top"]:.1f} 底={geom["d_bot"]:.1f} z={geom["height"]:.1f}nm'
        elif shape == 'trapezoid_pillar':
            lbl = f'd_bot={geom["d_bot"]:.1f} d_top={geom["d_top"]:.1f} h={geom["height"]:.1f}nm'
        else:
            lbl = f'd={geom["d_bot"]:.1f} z={geom["height"]:.1f}nm'

        ax2 = self.fig.add_subplot(gs[0, 1]); da(ax2, f'Synthetic GT  {lbl}')
        im2 = ax2.imshow(gt_al, cmap='afmhot', origin='upper', extent=ext)
        dcb(im2, ax2)

        avg_d = avg.copy()
        if shape in ('cylinder_hole', 'trapezoid_hole'):
            avg_d -= avg_d.max()
        else:
            avg_d -= avg_d.min()
        ax3 = self.fig.add_subplot(gs[0, 2]); da(ax3, f'Avg AFM Feature  (n={n})')
        im3 = ax3.imshow(avg_d, cmap='afmhot', origin='upper')
        dcb(im3, ax3)
        ax3.axhline(half, color='#f00', ls='--', lw=0.6, alpha=0.6)
        ax3.axvline(half, color='#f00', ls='--', lw=0.6, alpha=0.6)

        ax4 = self.fig.add_subplot(gs[1, 0]); da(ax4, 'Estimated Tip Shape')
        im4 = ax4.imshow(tip, cmap='RdBu_r', origin='upper', vmin=tip.min(), vmax=0)
        dcb(im4, ax4)
        ax4.axhline(half, color='w', ls='--', lw=0.5, alpha=0.4)
        ax4.axvline(half, color='w', ls='--', lw=0.5, alpha=0.4)

        try:
            cur_offset = float(self.offset_var.get())
        except (ValueError, tk.TclError):
            cur_offset = 0.0
        offset_tag = (f'下移{-cur_offset:.0f}nm' if cur_offset < 0 else
                      f'上移{cur_offset:.0f}nm'  if cur_offset > 0 else '原位')
        ax5 = self.fig.add_subplot(gs[1, 1])
        da(ax5, f'Cross-section  ({offset_tag})')
        x_nm = (np.arange(2 * half) - half) * px
        ax5.plot(x_nm, avg[half, :], color='#4f9cf9', lw=1.8,
                 label='AFM feature')
        ax5.plot(x_nm, gt_p[half, :],  color='#ef4444', lw=1.5, ls='--', label='GT ideal')
        # 真實底線/頂線，方便判斷藍線是否探底並與之平行
        if shape in ('cylinder_hole', 'trapezoid_hole'):
            ref_line, ref_lbl = gt_p.min(), '真實底線'
        else:
            ref_line, ref_lbl = gt_p.max(), '真實頂線'
        ax5.axhline(ref_line, color='#ef4444', ls=':', lw=1, alpha=0.5,
                    label=ref_lbl)
        ax5.plot(x_nm, tip[half, :],   color='#22c55e', lw=2,   label='Tip (reconstructed)')
        try:
            R  = float(self.tip_R_v.get() or 0)
            th = float(self.tip_angle_v.get() or 0)
            if R > 0 and th > 0:
                cone = make_cone_tip(half, px, R, th)
                # [FIX-10] cone 基線對齊：令頂點=0（向下為負），與其他曲線一致
                cone_prof = cone[half, :] - cone[half, :].max()
                ax5.plot(x_nm, cone_prof, color='#f59e0b', lw=1.5, ls=':',
                         label=f'Cone model R={R:.0f}nm θ={th:.0f}°')
        except Exception:
            pass
        ax5.axvline(0, color=C['border'], ls=':', lw=1)
        ax5.set_xlabel('nm', color=C['muted'], fontsize=7)
        ax5.set_ylabel('nm', color=C['muted'], fontsize=7)
        leg = ax5.legend(fontsize=6, facecolor=C['panel'], edgecolor=C['border'])
        [t.set_color(C['text']) for t in leg.get_texts()]
        ax5.grid(color=C['border'], alpha=0.5)

        # 右下：Rim 分析（進入率低時比 Tip 3D 更可靠）
        if rim is not None and not np.isnan(rim.get('rim_width_nm', float('nan'))):
            ax6 = self.fig.add_subplot(gs[1, 2])
            ax6.set_facecolor(C['panel'])
            rc = rim['r_centers']; rp = rim['profile']
            ax6.plot(rc, rp, color='#4f9cf9', lw=2, label='Radial avg')
            ax6.axhline(rp.max(), color='#ef4444', ls='--', lw=1, alpha=0.6, label='Surface')
            ax6.axhline(rp.min(), color='#f59e0b', ls='--', lw=1, alpha=0.6, label='Hole bottom')
            ax6.set_xlabel('r (nm)', color=C['muted'], fontsize=7)
            ax6.set_ylabel('Height (nm)', color=C['muted'], fontsize=7)
            entry_str = f'{rim["entry_rate"]*100:.0f}%'
            rim_str   = f'{rim["rim_width_nm"]:.1f}nm'
            cone_str  = f'{rim["cone_angle_deg"]:.1f}°' if not np.isnan(rim["cone_angle_deg"]) else 'N/A'
            ax6.set_title(f'Rim Analysis  (最可靠的資訊)\n進入率:{entry_str}  Rim寬:{rim_str}  錐角:{cone_str}',
                          color=C['warn'], fontsize=8, pad=3)
            ax6.tick_params(colors=C['muted'], labelsize=6)
            for sp in ax6.spines.values(): sp.set_edgecolor(C['border'])
            leg6 = ax6.legend(fontsize=7, facecolor=C['panel'], edgecolor=C['border'])
            [t.set_color(C['text']) for t in leg6.get_texts()]
            ax6.grid(color=C['border'], alpha=0.5)
        else:
            ax6 = self.fig.add_subplot(gs[1, 2], projection='3d')
            ax6.set_facecolor(C['panel'])
            X  = np.linspace(-half * px, half * px, 2 * half)
            Xg, Yg = np.meshgrid(X, X)
            ax6.plot_surface(Xg, Yg, tip, cmap='plasma', alpha=0.9, linewidth=0)
            ax6.set_xlabel('X(nm)', color=C['muted'], fontsize=6, labelpad=1)
            ax6.set_ylabel('Y(nm)', color=C['muted'], fontsize=6, labelpad=1)
            ax6.set_zlabel('nm',    color=C['muted'], fontsize=6, labelpad=1)
            ax6.tick_params(colors=C['muted'], labelsize=5)
            ax6.set_title('Tip 3D', color=C['text'], fontsize=9, pad=3)

        r_str     = f'{r_est:.0f}nm' if not np.isnan(r_est) else 'N/A'
        spec_note = ''
        try:
            R  = float(self.tip_R_v.get() or 0)
            th = float(self.tip_angle_v.get() or 0)
            if R > 0 and th > 0:
                cone   = make_cone_tip(half, px, R, th)
                c_half = half // 2
                t_in   = tip[half - c_half:half + c_half,
                             half - c_half:half + c_half].ravel()
                c_in   = cone[half - c_half:half + c_half,
                              half - c_half:half + c_half].ravel()
                corr = np.corrcoef(t_in, c_in)[0, 1]
                spec_note = f'  │  vs cone(R={R:.0f}nm,θ={th:.0f}°): corr={corr:.2f}'
        except Exception:
            pass

        self.fig.suptitle(
            f'Blind Tip Reconstruction  |  {os.path.basename(m["filepath"])}  '
            f'|  {shape}  |  est. tip radius: {r_str}{spec_note}',
            color=C['text'], fontsize=10, fontweight='bold')
        self.canvas.draw()

    # [FIX-1] 三元表達式加括號，消除跨行 SyntaxError
    def save_figure(self):
        """將目前畫布儲存為指定路徑的圖片"""
        afm_name = (
            os.path.splitext(os.path.basename(self.afm_path.get()))[0]
            if self.meta else 'afm_result'
        )
        p = filedialog.asksaveasfilename(
            title='儲存圖片',
            defaultextension='.png',
            initialfile=f'{afm_name}_result.png',
            filetypes=[('PNG 圖片', '*.png'), ('JPEG 圖片', '*.jpg'),
                       ('TIFF 圖片', '*.tif'), ('All files', '*.*')])
        if not p: return
        try:
            self.fig.savefig(p, dpi=150, facecolor=C['bg'], bbox_inches='tight')
            self.log(f'✓ 圖片儲存：{p}')
            messagebox.showinfo('完成', f'圖片已儲存至：\n{p}')
        except Exception as e:
            messagebox.showerror('失敗', str(e))
            self.log(f'✗ {e}')

    def save(self):
        if self.tip is None:
            messagebox.showwarning('提示', '請先執行重建')
            return
        p = filedialog.asksaveasfilename(
            title='儲存 tip.mat', defaultextension='.mat',
            initialfile='tip_estimated.mat',
            filetypes=[('MATLAB', '*.mat'), ('All', '*.*')])
        if not p: return
        try:
            savemat(p, {
                'tip':          self.tip,
                'px_nm':        self.meta['px_nm'],
                'shape_type':   self.shape_type,
                'd_bot_nm':     self.geom['d_bot'],
                'd_top_nm':     self.geom['d_top'],
                'height_nm':    self.geom['height'],
                'pitch_nm':     self.geom['pitch'],
                'scan_size_nm': self.meta['scan_nm'],
                'n_px':         self.meta['n_px'],
                'patch_half_px': self.tip_half,
            })
            png = p.replace('.mat', '_report.png')
            self.fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
            self.log(f'✓ {os.path.basename(p)} + report PNG')
            messagebox.showinfo('完成', f'已儲存：\n{p}\n\n報告：\n{png}')
        except Exception as e:
            messagebox.showerror('失敗', str(e))
            self.log(f'✗ {e}')

    # ── 廠商 cone 探針：用 ⑤ 的 R/θ 直接生成完整錐形探針並存成 tip.mat ──
    def _current_cone_tip(self):
        """依側欄 ⑤ 的 R/θ 生成 cone 探針陣列（頂點=0、旋轉對稱）。

        回傳 (cone, half, px, R, theta)；R/θ 無效時回傳 (None, ...)。
        cone 尺寸沿用重建時的 half（若尚未重建則用 auto_params 推算），
        與重建 tip 同 px_nm、同陣列慣例，可直接餵給訓練 pipeline。
        """
        if self.meta is None:
            return None, None, None, None, None
        try:
            R     = float(self.tip_R_v.get() or 0)
            theta = float(self.tip_angle_v.get() or 0)
        except ValueError:
            return None, None, None, None, None
        if R <= 0 or theta <= 0:
            return None, None, None, None, None
        px   = self.meta['px_nm']
        half = self.tip_half
        if half is None:                       # 尚未重建 → 用幾何自動推算 half
            geom = self._get_geom()
            half = auto_params(geom, self.shape_var.get(), self.meta)['half']
        cone = make_cone_tip(half, px, R, theta)   # 頂點=0、向外為負、旋轉對稱
        return cone, half, px, R, theta

    def save_cone(self):
        """把廠商 cone 模型存成 tip.mat（保留重建功能，另存一份 cone 探針）。"""
        cone, half, px, R, theta = self._current_cone_tip()
        if cone is None:
            messagebox.showwarning(
                '提示',
                '請先載入 AFM 檔並在「⑤ 探針廠商規格」填入\n'
                '有效的 尖端半徑 R 與 半錐角 θ（皆 > 0）')
            return
        p = filedialog.asksaveasfilename(
            title='儲存 Cone tip.mat', defaultextension='.mat',
            initialfile=f'tip_cone_R{R:.0f}_th{theta:.0f}.mat',
            filetypes=[('MATLAB', '*.mat'), ('All', '*.*')])
        if not p: return
        try:
            geom = self.geom if self.geom else self._get_geom()
            savemat(p, {
                'tip':          cone,
                'px_nm':        px,
                'tip_source':   'vendor_cone',
                'cone_R_nm':    R,
                'cone_theta_deg': theta,
                'shape_type':   self.shape_type or self.shape_var.get(),
                'd_bot_nm':     geom['d_bot'],
                'd_top_nm':     geom['d_top'],
                'height_nm':    geom['height'],
                'pitch_nm':     geom['pitch'],
                'scan_size_nm': self.meta['scan_nm'],
                'n_px':         self.meta['n_px'],
                'patch_half_px': half,
            })
            self.log(f'✓ Cone tip 已儲存：{os.path.basename(p)} '
                     f'(R={R:.0f}nm θ={theta:.0f}° half={half}px)')
            messagebox.showinfo(
                '完成',
                f'已儲存廠商 cone 探針：\n{p}\n\n'
                f'R={R:.1f}nm  θ={theta:.1f}°  尺寸={2*half}×{2*half}px')
        except Exception as e:
            messagebox.showerror('失敗', str(e))
            self.log(f'✗ {e}')

    def on_close(self):
        plt.close('all')
        self.destroy()


if __name__ == '__main__':
    App().mainloop()