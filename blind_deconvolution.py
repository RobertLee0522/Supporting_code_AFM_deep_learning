#!/usr/bin/env python3
"""
blind_deconvolution.py — 通用 AFM 探針去卷積引擎（零訓練，原型）

不需要深度學習、不需要為每支探針/每個樣品重新訓練。基於 Villarrubia (1997)
形態學方法（Gwyddion 等商用軟體採用的標準）：

  成像（前向）  i = s ⊕ t        探針對真實表面做 grey dilation
  還原（去卷積）s_r = i ⊖ t       用探針對影像做 grey erosion → 表面「確定下界」
  certainty     標出探針『實際接觸過』的點 → 其餘是碰不到的死角（不可信、不腦補）

為什麼這條路才是通用產品的核心（相對於「為單一探針訓練模型」）：
  • 探針當「執行時輸入」而非燒進權重 → 換任何廠牌探針都適用，不需重訓。
  • 數學嚴謹：還原滿足 s ≤ s_r ≤ i 的「確定下界」，不會腦補（不會過度去卷積）。
  • 誠實：certainty map 明確標出「探針碰不到、無法驗證」的區域。

探針來源（本引擎吃『已知探針』，不自己盲估）：
  (A) 廠商 cone 規格 —— 用 make_cone_tip(R, θ) 生成（最穩健，本專案推薦）。
  (B) 用尖刺校正光柵（TGT1 等）在 **Gwyddion** 跑 blind tip estimation
      （itip_estimate，已驗證實作），匯出 tip 後存成 .npy 餵進來。
  ⚠ 本檔不自行實作 blind 盲估 —— 該演算法（Villarrubia 迭代）需仔細實作與驗證，
     用 Gwyddion 的成熟版本比我重造一個易錯的更可靠、更誠實。

探針對稱性：真實探針未必旋轉對稱。本引擎支援
  • 對稱 cone（單一半錐角 θ）
  • 非對稱 cone（θx / θy 兩軸各自角度，橢圓內插；make_cone_tip_asym）
  • 或直接載入任意 2D 探針 .npy（Gwyddion 匯出）。
形態學 erosion/certainty 對任意 2D 探針皆成立，非對稱無需改動核心。

用法：
  python blind_deconvolution.py                              # 預設啟動圖形介面（GUI）
  python blind_deconvolution.py --gui                        # 明確啟動 GUI（步驟式操作）
  python blind_deconvolution.py --demo                       # 合成資料驗證 + 教學圖
  python blind_deconvolution.py scan.npy --tip tip.npy       # 用已知探針去卷積
  python blind_deconvolution.py scan.npy --cone-R 2 --cone-theta 25 --tip-half 5
                                                             # 用廠商 cone 去卷積
  python blind_deconvolution.py std.000 --cone-R 2 --sample hole
                                                             # 直接開 Nanoscope .000 回推
（輸入可為 Nanoscope .000 原始檔（自動解析/去尖刺/去傾斜/帶入 px_nm），
  或 nm 單位 2D .npy 陣列）
"""
import os
import sys
import re
import argparse
# Windows 主控台預設 cp950 無法輸出 ≤/⊕ 等符號，改用 UTF-8 避免 print 崩潰（不影響 GUI）
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # 無頭環境輸出 PNG（GUI 以 FigureCanvasTkAgg 嵌入，不受此影響）
# ── Matplotlib CJK 字型（讓圖中中文正確顯示）────────────────────────
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'Microsoft YaHei',   # Windows
    'PingFang TC', 'PingFang SC', 'Heiti TC',  # macOS
    'Noto Sans CJK TC', 'Noto Sans CJK SC',    # Linux
    'WenQuanYi Micro Hei', 'SimHei',
] + matplotlib.rcParams.get('font.sans-serif', [])
matplotlib.rcParams['axes.unicode_minus'] = False  # 負號不用方塊替代
import matplotlib.pyplot as plt
from scipy.ndimage import (grey_dilation, grey_erosion, median_filter, label,
                           map_coordinates)


# ──────────────────────────────────────────────────────────────────
# 核心形態學運算
#   探針慣例：apex 在中心 = 0，向外為負（與 afm_gui / Supporting_code 一致）。
#   探針旋轉對稱 → scipy 的 reflect 慣例不影響結果。
# ──────────────────────────────────────────────────────────────────
def image_from_surface(surface, tip):
    """前向成像：i = grey_dilation(s, tip)。與訓練 pipeline 一致（孔變淺/變窄）。"""
    return grey_dilation(surface, structure=tip, mode='nearest')


def reconstruct_surface(image, tip):
    """certainty 去卷積：s_r = grey_erosion(i, tip)。

    morphological 對偶保證 s ≤ s_r ≤ i：
      • s_r 比影像 i 更接近真實（孔更深/更寬）；
      • 但仍是「下界」—— 探針碰不到處只還原到下界，**不會超過 → 不會過度去卷積**。
    """
    return grey_erosion(image, structure=tip, mode='nearest')


def surface_certainty(image, tip, rel_tol=1e-3):
    """標出『探針實際接觸過』的表面點（certain），其餘為碰不到的死角。

    原理：把還原表面 s_r 重新成像 i_re=dilation(s_r,t)；在 i_re==i 的影像點 x，
    探針 apex 確實接觸，其接觸的表面點 = argmax_v[s_r(x+v)+t(v)]，標為 certain。
    未被任何接觸點命中的表面區域（如寬孔平底）→ uncertain（只有下界，不可信）。

    回傳 (recon, certain_mask, certain_frac)。
    """
    recon = reconstruct_surface(image, tip)
    H, W  = image.shape
    h     = tip.shape[0] // 2
    tol   = rel_tol * (float(np.abs(image).max()) + 1e-9) + 1e-6

    certain = np.zeros((H, W), dtype=bool)
    pad = np.pad(recon, h, mode='edge')                  # 邊界以邊緣值延伸
    for di in range(-h, h + 1):                          # 對每個探針位移累積候選
        for dj in range(-h, h + 1):
            # cand[x] = s_r(x+v) + t(v)，v=(di,dj)
            cand = pad[h + di:h + di + H, h + dj:h + dj + W] + tip[h + di, h + dj]
            if di == -h and dj == -h:
                best = cand.copy(); best_di = np.full((H, W), di); best_dj = np.full((H, W), dj)
            else:
                upd = cand > best
                best = np.where(upd, cand, best)
                best_di = np.where(upd, di, best_di)
                best_dj = np.where(upd, dj, best_dj)
    # best == i_re（重新成像）；contact 影像點：i_re ≈ i
    contact = np.abs(best - image) <= tol
    cy, cx = np.where(contact)
    ty = np.clip(cy + best_di[cy, cx], 0, H - 1)         # 接觸到的表面點
    tx = np.clip(cx + best_dj[cy, cx], 0, W - 1)
    certain[ty, tx] = True
    return recon, certain, float(certain.mean())


def deconvolve(image, tip):
    """便利封裝：回傳 dict(recon, certain, certain_frac)。"""
    recon, certain, frac = surface_certainty(image, tip)
    return {'recon': recon, 'certain': certain, 'certain_frac': frac}


# ──────────────────────────────────────────────────────────────────
# 探針生成（廠商 cone / 拋物面）—— 供去卷積與 demo 使用
# ──────────────────────────────────────────────────────────────────
def make_cone_tip(half_px, px_nm, R_nm=2.0, theta_deg=25.0):
    """球冠+直線錐壁探針（apex=0、向外為負），與 afm_gui/Supporting_code 同公式。"""
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


def make_cone_tip_asym(half_px, px_nm, R_nm=2.0, theta_x_deg=25.0, theta_y_deg=15.0):
    """非對稱錐探針：X/Y 兩軸各自的半錐角（θx, θy），球冠+直線錐壁、apex=0、向外為負。

    物理：真實探針常非旋轉對稱（例如做成刀刃狀或製程不對稱），沿快軸/慢軸的
    有效錐角不同。此處把「半錐角隨方位角 φ 變化」建成橢圓內插——
      tan θ(φ) = (a·b) / sqrt((b·cosφ)² + (a·sinφ)²)，a=tanθx, b=tanθy
    使 φ=0（沿 X）得 θx、φ=90°（沿 Y）得 θy，兩軸間平滑過渡。
    當 θx==θy 時退化為對稱 cone（與 make_cone_tip 一致）。
    """
    size = 2 * half_px + 1
    yi, xi = np.indices((size, size))
    dy = (yi - half_px).astype(float)
    dx = (xi - half_px).astype(float)
    r  = np.sqrt(dy ** 2 + dx ** 2) * px_nm
    phi = np.arctan2(dy, dx)
    a, b = np.tan(np.radians(theta_x_deg)), np.tan(np.radians(theta_y_deg))
    tan_th = (a * b) / np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2 + 1e-12)
    rt = R_nm * (tan_th / np.sqrt(1.0 + tan_th ** 2))          # = R·sinθ（每 px 方向不同）
    tip = np.zeros((size, size))
    cap = r <= rt                                             # 球冠區
    tip[cap] = -(R_nm - np.sqrt(np.maximum(R_nm ** 2 - r[cap] ** 2, 0.0)))
    zt = -(R_nm - np.sqrt(np.maximum(R_nm ** 2 - rt ** 2, 0.0)))
    wall = ~cap                                              # 直線錐壁區
    tip[wall] = zt[wall] - (r[wall] - rt[wall]) / tan_th[wall]
    tip -= tip.max()
    return tip


def make_paraboloid_tip(half_px, px_nm, R_nm=46.9):
    """拋物面探針 z(r) = −r²/(2R)（apex=0）。較鈍 → 明顯失真孔洞，適合 demo。"""
    size = 2 * half_px + 1
    yi, xi = np.indices((size, size))
    r = np.sqrt((yi - half_px) ** 2 + (xi - half_px) ** 2) * px_nm
    tip = -(r ** 2) / (2.0 * R_nm)
    tip -= tip.max()
    return tip


def auto_tip_half(surface, px_nm, R_nm, theta_deg, sample='hole', cap=60):
    """依『載入影像量得的特徵起伏』+ 探針 R/θ 自動換算視窗半徑（px）。

    物理：深度 D 的孔（或高 h 的凸起）被半錐角 θ、球冠 R 的探針掃過時，
    接觸點側向延伸 ≈ R·sinθ + D·tanθ（nm）。視窗需 ≥ 此側向延伸再留 1px 邊界。
      • px_nm、D 兩個跟樣品有關的量 → 從 .000 檔（Scan Size、高度資料）取得；
      • R、θ 為探針物理規格 → 檔案裡沒有，需由 datasheet 提供。
    θ 為『半錐角(half cone angle)』；若手上是全開角(total angle)請先除以 2。
    非對稱探針請以較大的一軸角度傳入（涵蓋較寬方向）。
    """
    if sample == 'bump':                         # 凸起：表面為低值、特徵向上
        amp = float(np.percentile(surface, 99.5) - np.percentile(surface, 10))
    else:                                        # 孔洞：表面為高值、特徵向下
        amp = float(np.percentile(surface, 95) - surface.min())
    th = np.radians(theta_deg)
    r_nm = R_nm * np.sin(th) + max(amp, 0.0) * np.tan(th)
    return int(min(cap, max(3, np.ceil(r_nm / px_nm) + 1)))


# ──────────────────────────────────────────────────────────────────
# 特徵寬度量測（還原表面 vs 影像對照）
# ──────────────────────────────────────────────────────────────────
def _half_span_bounds(line, i0, thr):
    """回傳門檻 thr 在極值索引 i0 兩側的『亞像素』跨越邊界 (left, right, width_px)。

    自 i0 向左/右擴張到低於 thr 的相鄰點，再對跨越邊做線性內插取亞像素位置。
    """
    n = len(line)
    if line[i0] < thr:
        return float(i0), float(i0), 0.0
    L = i0
    while L > 0 and line[L - 1] >= thr:
        L -= 1
    if L > 0 and line[L] != line[L - 1]:
        left = (L - 1) + (thr - line[L - 1]) / (line[L] - line[L - 1])
    else:
        left = float(L)
    R = i0
    while R < n - 1 and line[R + 1] >= thr:
        R += 1
    if R < n - 1 and line[R] != line[R + 1]:
        right = R + (line[R] - thr) / (line[R] - line[R + 1])
    else:
        right = float(R)
    return left, right, max(0.0, right - left)


def measure_feature_width(surface, px_nm, sample='hole', frac=0.5,
                          at=None, search_px=12):
    """量測表面特徵的寬度。

    步驟：以表面基線為 0 把特徵轉成正向振幅（孔洞=深度、凸起=高度）→ 找極值點 →
    沿該點 X、Y 剖面量在 `frac×振幅` 門檻的連續跨距（FWHM，亞像素內插）
    → 另以門檻連通區面積算等效直徑 ⌀=2√(A/π)。回傳 nm 單位 dict；無特徵回 None。

    frac=0.5 即半深/半高全寬（FWHM，AFM 常用）；可調 0.05–0.95。
    at=(y, x)：**指定量測位置**（如 GUI 點選）——在其 ±search_px 視窗內找局部
    極值當特徵中心，門檻用該特徵『自己的振幅』（局部量測，不受畫面最大特徵影響）；
    at=None 則自動取全圖最深/最高特徵。
    """
    surf = np.asarray(surface, dtype=float)
    if sample == 'bump':                          # 凸起：表面低值、特徵向上
        feat = surf - np.percentile(surf, 10)
    else:                                         # 孔洞：表面高值，轉成深度為正
        feat = np.percentile(surf, 95) - surf
    H, W = feat.shape
    if at is not None:                            # 點選模式：視窗內局部極值
        ay = int(np.clip(round(at[0]), 0, H - 1))
        ax_ = int(np.clip(round(at[1]), 0, W - 1))
        ys, ye = max(0, ay - search_px), min(H, ay + search_px + 1)
        xs, xe = max(0, ax_ - search_px), min(W, ax_ + search_px + 1)
        dy, dx = np.unravel_index(int(np.argmax(feat[ys:ye, xs:xe])),
                                  (ye - ys, xe - xs))
        gy, gx = ys + dy, xs + dx
        amp = float(feat[gy, gx])                 # 該特徵自己的振幅
    else:                                         # 自動模式：全圖極值
        gy, gx = np.unravel_index(int(np.argmax(feat)), feat.shape)
        amp = float(feat.max())
    if amp <= 1e-9:
        return None
    thr = frac * amp
    # 取『含極值點』的連通區為主特徵，量測經其『形心』的 X/Y 剖面
    # （直接用 argmax 會落在平底特徵的邊角，剖面量到短弦 → 錯誤，故改用形心）
    lbl, n = label(feat >= thr)
    if n == 0 or lbl[gy, gx] == 0:
        return None
    comp = lbl == lbl[gy, gx]
    ys, xs = np.where(comp)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    x0, x1, wx = _half_span_bounds(feat[cy, :], cx, thr)
    y0, y1, wy = _half_span_bounds(feat[:, cx], cy, thr)
    area_px = int(comp.sum())
    return {'cx': cx, 'cy': cy, 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
            'width_x_nm': wx * px_nm, 'width_y_nm': wy * px_nm,
            'equiv_diam_nm': 2.0 * np.sqrt(area_px / np.pi) * px_nm,
            'amp_nm': amp, 'frac': frac}


def make_hole_surface(N=128, px_nm=39.1, open_nm=228.8, bot_nm=183.5, depth=125.8):
    """寬孔梯形樣品（本專案樣品）。回傳真實表面（供 demo 對照）。"""
    yy, xx = np.indices((N, N))
    s = np.zeros((N, N))
    for cy in range(24, N, 40):
        for cx in range(24, N, 40):
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * px_nm
            ro, ri = open_nm / 2, bot_nm / 2
            s[r <= ri] = -depth
            wall = (r > ri) & (r <= ro)
            s[wall] = -depth * (1 - (r[wall] - ri) / (ro - ri))
    return s


def depth_of(arr):
    return float(np.percentile(arr, 95) - arr.min())


# ──────────────────────────────────────────────────────────────────
# Nanoscope 原始檔載入（TF-free，沿用 detect.py / reconstruct_lines_3d 邏輯）
#   讓 GUI/CLI 直接開 .000 等原始掃描檔回推，不需先轉 .npy。
# ──────────────────────────────────────────────────────────────────
def is_nanoscope_file(path):
    """副檔名以 3 位數字開頭（.000/.004…，容許後接描述字，如 .000-after_flatten）。"""
    return bool(re.match(r'^\.\d{3}', os.path.splitext(path)[1]))


def _parse_nanoscope_header(fp):
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
            chs.append({'offset': int(off.group(1)),
                        'bpp': int(bpp.group(1)) if bpp else 2,
                        'data_length': int(dln.group(1)) if dln else None,
                        'name': nm.group(2).strip(),
                        'z_key': zsc.group(1) if zsc else 'ZsensSens',
                        'z_lsb': float(zsc.group(2)) if zsc else 0.000375})
    return {'n_px': n_px, 'n_lines': n_ln, 'scan_nm': scan, 'px_nm': scan / n_px,
            'zsens': zs, 'zsens_s': zss, 'channels': chs}


def _read_height_channel(fp, meta):
    """讀高度通道 → nm 2D array，以 Data length 為界防跨通道污染（detect.py 邏輯）。"""
    ch_idx = 0
    for i, ch in enumerate(meta['channels']):
        if any(k in ch['name'].lower() for k in ('zsensor', 'height sensor', 'height')):
            ch_idx = i
            break
    ch = meta['channels'][ch_idx]
    dt = np.int16 if ch['bpp'] == 2 else np.int32
    read_bytes = meta['n_px'] * meta['n_lines'] * ch['bpp']
    if ch.get('data_length'):
        read_bytes = min(read_bytes, ch['data_length'])
    with open(fp, 'rb') as f:
        f.seek(ch['offset'])
        raw = np.frombuffer(f.read(read_bytes), dtype=dt)
    actual = len(raw) // meta['n_px']
    if actual == 0:
        raise ValueError('通道資料不足，無法組成影像')
    img  = raw[:actual * meta['n_px']].reshape(actual, meta['n_px']).astype(np.float64)
    sens = meta['zsens_s'] if 'ZsensSens' in ch['z_key'] else meta['zsens']
    return img * (ch['z_lsb'] * sens), ch['name'], actual


def _despike(img, win=3, abs_thresh_nm=300.0, n_sigma=8.0):
    """去孤立尖刺（中值殘差門檻）；真實深孔/壞線鄰域同偏移故保留。"""
    med   = median_filter(img, size=win)
    resid = img - med
    mad   = np.median(np.abs(resid - np.median(resid)))
    thr   = max(abs_thresh_nm, n_sigma * 1.4826 * mad)
    m     = np.abs(resid) > thr
    if m.any():
        img = img.copy(); img[m] = med[m]
    return img, int(m.sum())


def _flatten_rows(img, sample='hole'):
    """逐行一階去傾斜，再以百分位基線歸零。孔洞:表面為高值(pct95)、凸起:低值(pct10)。
    （erosion 對垂直平移不變，基線只影響顯示/certainty，不改變還原形狀。）"""
    flat = img.astype(np.float64).copy()
    x = np.arange(img.shape[1])
    for i in range(img.shape[0]):
        flat[i] -= np.polyval(np.polyfit(x, flat[i], 1), x)
    return flat - np.percentile(flat, 95 if sample == 'hole' else 10)


def load_nanoscope_surface(fp, sample='hole'):
    """載入 Nanoscope 原始掃描 → nm 單位 2D 表面。

    流程：解析標頭 → 讀高度通道(Z 校正) → 去尖刺 → (孔洞另做物理裁切移除連續壞區)
    → 逐行去傾斜。保留原生解析度（不 resize），使影像 px_nm 與探針一致。
    回傳 (surface_nm, px_nm, info_str)。
    """
    meta = _parse_nanoscope_header(fp)
    img, ch_name, actual = _read_height_channel(fp, meta)
    img, n_spike = _despike(img)
    if sample == 'hole':                       # 孔洞表面為高基準，收緊高側移除大台階
        ref = float(np.median(img))
        img = np.clip(img, ref - 300.0, ref + 60.0)
    flat = _flatten_rows(img, sample)
    info = (f"ch='{ch_name}' {flat.shape} px_nm={meta['px_nm']:.2f} "
            f"尖刺={n_spike} 行={actual}/{meta['n_lines']}")
    return flat, meta['px_nm'], info


# ──────────────────────────────────────────────────────────────────
# 視覺化
# ──────────────────────────────────────────────────────────────────
def _draw_width(ax, meas, color='red'):
    """在給定 axes 上畫寬度量測：X/Y 跨距線 + 極值點 + 數值標註（color 區分影像/還原）。"""
    m = meas
    ax.plot([m['x0'], m['x1']], [m['cy'], m['cy']], '-', color=color, lw=1.6)
    ax.plot([m['cx'], m['cx']], [m['y0'], m['y1']], '-', color=color, lw=1.6)
    ax.plot(m['cx'], m['cy'], '+', color=color, ms=9, mew=1.6)
    ax.text(0.02, 0.98,
            f"W@{int(m['frac']*100)}%\nX={m['width_x_nm']:.1f}nm\n"
            f"Y={m['width_y_nm']:.1f}nm\nD={m['equiv_diam_nm']:.1f}nm",
            transform=ax.transAxes, va='top', ha='left', fontsize=8, color='white',
            bbox=dict(boxstyle='round', fc=color, ec='none', alpha=0.65))


def save_panels(image, recon, certain, tip, certain_frac, out_path, title='',
                meas=None, meas_in=None):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    vmin = min(image.min(), recon.min())
    vmax = max(image.max(), recon.max())

    im0 = ax[0].imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title(f'Input image (AFM)\nmin={image.min():.1f} max={image.max():.1f} nm')
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    if meas_in:                                        # 影像寬度（橘色，對照用）
        _draw_width(ax[0], meas_in, color='#ff8c00')

    im1 = ax[1].imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title(f'Reconstructed surface (erosion)\n'
                    f'min={recon.min():.1f} (certain lower bound)')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    if meas:                                            # 疊上寬度量測線與數值
        _draw_width(ax[1], meas)

    ax[2].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
    overlay = np.zeros((*certain.shape, 4))
    overlay[~certain] = [1, 0, 0, 0.40]                 # 紅 = 探針碰不到、不可信
    ax[2].imshow(overlay)
    ax[2].set_title(f'Certainty map\ncertain={certain_frac*100:.1f}%  '
                    f'(red = tip never touched)')

    half = tip.shape[0] // 2
    ax[3].plot(np.arange(-half, half + 1), tip[half, :], color='#1d9e75', lw=2)
    ax[3].axhline(0, color='gray', lw=0.6)
    ax[3].set_title('Tip profile (apex=0)')
    ax[3].set_xlabel('px'); ax[3].set_ylabel('nm'); ax[3].grid(alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  [圖片] {out_path}")


# ──────────────────────────────────────────────────────────────────
# Demo：合成驗證 + 教學
# ──────────────────────────────────────────────────────────────────
def run_demo(out_dir='blind_demo'):
    os.makedirs(out_dir, exist_ok=True)
    px_nm = 39.1
    truth = make_hole_surface(px_nm=px_nm)

    print("=" * 66)
    print("Demo：用『已知鈍探針』做 certainty 去卷積（驗證正確性 + 誠實死角）")
    print("=" * 66)
    # 較鈍的拋物面探針：會明顯失真孔洞，才看得出還原與不可信死角
    tip = make_paraboloid_tip(half_px=5, px_nm=px_nm, R_nm=46.9)
    image = image_from_surface(truth, tip)
    res = deconvolve(image, tip)
    recon, certain, frac = res['recon'], res['certain'], res['certain_frac']

    # 1) 下界性質（嚴謹性）
    ok = bool(np.all(recon >= truth - 1e-6) and np.all(recon <= image + 1e-6))
    print(f"  下界性質 s ≤ s_r ≤ i 成立：{ok}（保證不會過度去卷積）")

    # 2) 還原品質：橫向孔徑（影像被探針 narrow → 還原應變回較寬）
    def width_at(arr, frac_d=0.5):
        base = np.percentile(arr, 95); d = base - arr; mx = d.max()
        return 2 * np.sqrt((d > frac_d * mx).sum() / np.pi)  # 粗略等效像素徑
    print(f"  孔徑(等效px @50%深)：真實={width_at(truth):.1f}  影像={width_at(image):.1f}  "
          f"還原={width_at(recon):.1f}")
    print(f"     → 還原孔徑介於『影像(被narrow)』與『真實』之間，往真值靠近但不超過")

    # 3) certainty：探針碰不到的孔底死角應被標為 uncertain
    print(f"  可信像素 certain={frac*100:.1f}%（紅=探針碰不到，誠實標示）")
    # 量「不可信」主要落在孔底（深處）
    deep = (np.percentile(truth, 95) - truth) > 0.8 * (np.percentile(truth,95)-truth.min())
    if deep.sum():
        print(f"     孔底深處被標為 uncertain 的比例："
              f"{100*(~certain)[deep].mean():.0f}%（應偏高=底部資訊遺失）")

    save_panels(image, recon, certain, tip, frac,
                os.path.join(out_dir, 'demo_certainty_deconv.png'),
                title='Certainty deconvolution (known tip) — honest lower bound')
    print(f"\n完成。教學圖：{out_dir}/demo_certainty_deconv.png")
    print("解讀：紅色死角=探針物理上碰不到、無法驗證的區域（如寬孔平底），")
    print("      引擎只給『確定下界』、不腦補 —— 這正是相對於 DL 過度去卷積的優勢。")


# ──────────────────────────────────────────────────────────────────
# 圖形介面（Tkinter）：讓使用者「按流程」操作
#   ① 載入影像(.npy) → ② 設定 px_nm → ③ 設定探針（對稱 θ／非對稱 θx,θy 或載入探針）
#   → ④ 執行去卷積 → ⑤ 檢視 certainty → ⑥ 儲存結果
# 形態學運算對任意 2D 探針皆成立，非對稱探針無需改動核心（reconstruct/certainty）。
# ──────────────────────────────────────────────────────────────────
def _fnum(s):
    """寬容數字解析：接受中文輸入法的全形小數點（。．，）與逗號。

    使用者以中文 IME 輸入「20。7」時 float() 會失敗，看起來像「不能輸入小數點」；
    此處統一正規化成半形句點再解析。
    """
    return float(str(s).strip()
                 .replace('。', '.').replace('．', '.')
                 .replace('，', '.').replace(',', '.'))


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import threading
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401 (註冊 3d 投影)

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title('AFM 通用去卷積工具（零訓練 · Villarrubia certainty）')
            self.geometry('1200x780')
            # 狀態
            self.image = None          # 輸入影像 (nm 2D)
            self.image_path = None
            self.tip = None            # 目前探針
            self.tip_npy_path = None   # 載入的探針檔
            self.recon = None
            self.certain = None
            self.frac = None
            self.meas = None           # 還原表面寬度量測結果
            self.meas_in = None        # 影像寬度量測結果（對照）
            self.pick = None           # 使用者點選的量測位置 (y, x)；None=自動
            self._auto_job = None      # 參數變更自動重跑（防抖）排程 id
            self._meas_job = None      # 半深比例變更自動重量測排程 id
            self._busy = False         # 去卷積執行中旗標（避免並行重跑）
            self._last_run_key = None  # 上次去卷積的參數指紋（相同就不重跑）
            # NanoScope Section 式剖面線：p0/p1=端點(y,x)、c1/c2=游標參數位置(0~1)
            self.section = None
            self._drag = None          # 進行中的拖曳 ('newline'|'endpoint'|'cursor'|'pick')
            self._sa = {}              # 剖面線 artists（拖曳時輕量更新用）
            self._build_ui()

        # ── 版面 ──────────────────────────────────────────────
        def _build_ui(self):
            left = ttk.Frame(self, padding=8)
            left.pack(side='left', fill='y')
            right = ttk.Frame(self)
            right.pack(side='right', fill='both', expand=True)

            def step(title):
                lf = ttk.LabelFrame(left, text=title, padding=6)
                lf.pack(fill='x', pady=4)
                return lf

            # ① 影像
            s1 = step('① 載入掃描檔')
            ttk.Button(s1, text='📂 選擇掃描檔 (.000 原始檔 或 .npy)',
                       command=self._load_image).pack(fill='x')
            self.lbl_img = ttk.Label(s1, text='（尚未載入）', wraplength=250,
                                     foreground='#666')
            self.lbl_img.pack(fill='x', pady=(4, 0))
            srow = ttk.Frame(s1); srow.pack(fill='x', pady=(4, 0))
            ttk.Label(srow, text='樣品類型：').pack(side='left')
            self.var_sample = tk.StringVar(value='hole')
            ttk.Radiobutton(srow, text='孔洞', value='hole',
                            variable=self.var_sample,
                            command=self._on_sample_change).pack(side='left')
            ttk.Radiobutton(srow, text='凸起', value='bump',
                            variable=self.var_sample,
                            command=self._on_sample_change).pack(side='left')
            ttk.Label(s1, text='※ .000 原始檔會自動去尖刺+去傾斜並帶入 px_nm；'
                              '樣品類型只影響基線方向，不改變 erosion 還原結果',
                      wraplength=250, foreground='#999',
                      font=('', 8)).pack(fill='x')

            # ② 尺度
            s2 = step('② 影像尺度')
            row = ttk.Frame(s2); row.pack(fill='x')
            ttk.Label(row, text='每 px = ').pack(side='left')
            self.var_px = tk.StringVar(value='39.1')
            ttk.Entry(row, textvariable=self.var_px, width=8).pack(side='left')
            ttk.Label(row, text=' nm').pack(side='left')

            # ③ 探針
            s3 = step('③ 探針設定')
            self.var_source = tk.StringVar(value='cone')
            ttk.Radiobutton(s3, text='廠商 cone（輸入規格）', value='cone',
                            variable=self.var_source,
                            command=self._update_tip_mode).pack(anchor='w')
            ttk.Radiobutton(s3, text='載入探針檔（Gwyddion itip_estimate .npy）',
                            value='npy', variable=self.var_source,
                            command=self._update_tip_mode).pack(anchor='w')

            # cone 參數
            cone = ttk.Frame(s3); cone.pack(fill='x', padx=(14, 0), pady=2)
            r1 = ttk.Frame(cone); r1.pack(fill='x')
            ttk.Label(r1, text='球冠半徑 R = ').pack(side='left')
            self.var_R = tk.StringVar(value='2.0')
            self.e_R = ttk.Entry(r1, textvariable=self.var_R, width=7)
            self.e_R.pack(side='left'); ttk.Label(r1, text=' nm').pack(side='left')

            r2 = ttk.Frame(cone); r2.pack(fill='x')
            ttk.Label(r2, text='視窗半徑 = ').pack(side='left')
            self.var_half = tk.StringVar(value='5')
            self.e_half = ttk.Entry(r2, textvariable=self.var_half, width=6)
            self.e_half.pack(side='left'); ttk.Label(r2, text=' px').pack(side='left')
            self.var_autohalf = tk.BooleanVar(value=True)
            ttk.Checkbutton(r2, text='自動', variable=self.var_autohalf,
                            command=self._update_tip_mode).pack(side='left', padx=(6, 0))
            ttk.Label(cone, text='（自動 = 依載入孔深 + R/θ 換算；θ 為半錐角）',
                      foreground='#999', font=('', 8)).pack(anchor='w')

            # 對稱 / 非對稱
            self.var_asym = tk.BooleanVar(value=False)
            self.chk_asym = ttk.Checkbutton(
                cone, text='☑ 非對稱探針（θx / θy 分開設定）',
                variable=self.var_asym, command=self._update_tip_mode)
            self.chk_asym.pack(anchor='w', pady=(4, 0))

            r3 = ttk.Frame(cone); r3.pack(fill='x')
            ttk.Label(r3, text='半錐角 θ = ').pack(side='left')
            self.var_th = tk.StringVar(value='25')
            self.e_th = ttk.Entry(r3, textvariable=self.var_th, width=7)
            self.e_th.pack(side='left'); ttk.Label(r3, text=' °（對稱）').pack(side='left')

            r4 = ttk.Frame(cone); r4.pack(fill='x')
            ttk.Label(r4, text='θx = ').pack(side='left')
            self.var_thx = tk.StringVar(value='25')
            self.e_thx = ttk.Entry(r4, textvariable=self.var_thx, width=6)
            self.e_thx.pack(side='left')
            ttk.Label(r4, text=' °   θy = ').pack(side='left')
            self.var_thy = tk.StringVar(value='15')
            self.e_thy = ttk.Entry(r4, textvariable=self.var_thy, width=6)
            self.e_thy.pack(side='left'); ttk.Label(r4, text=' °').pack(side='left')

            # 載入探針檔
            self.btn_tipnpy = ttk.Button(s3, text='📂 選擇探針 .npy',
                                         command=self._load_tip_npy)
            self.btn_tipnpy.pack(fill='x', pady=(4, 0))
            self.lbl_tip = ttk.Label(s3, text='', wraplength=250, foreground='#666')
            self.lbl_tip.pack(fill='x')

            ttk.Button(s3, text='👁 預覽探針形狀',
                       command=self._preview_tip).pack(fill='x', pady=(4, 0))

            # ④ 執行
            s4 = step('④ 執行')
            ttk.Button(s4, text='▶ 執行去卷積', command=self._run).pack(fill='x')

            # ⑤ 結果
            s5 = step('⑤ 結果')
            g = ttk.Frame(s5); g.pack(fill='x')
            self.lbl_cert = self._stat(g, '可信像素 certain', 0)
            self.lbl_din = self._stat(g, '影像起伏', 1)
            self.lbl_dre = self._stat(g, '還原起伏（下界）', 2)
            self.lbl_win = self._stat(g, '影像寬度 ⌀', 3)
            self.lbl_wre = self._stat(g, '還原寬度', 4)
            # 寬度量測控制
            mrow = ttk.Frame(s5); mrow.pack(fill='x', pady=(4, 0))
            ttk.Label(mrow, text='半深比例 ').pack(side='left')
            self.var_wfrac = tk.StringVar(value='0.5')
            ttk.Entry(mrow, textvariable=self.var_wfrac, width=5).pack(side='left')
            ttk.Button(mrow, text='📏 量測寬度',
                       command=self._measure).pack(side='left', padx=(6, 0))
            ttk.Button(mrow, text='↺ 自動',
                       command=self._pick_reset).pack(side='left', padx=(4, 0))
            ttk.Button(mrow, text='✕ 剖面線',
                       command=self._section_clear).pack(side='left', padx=(4, 0))
            self.lbl_pick = ttk.Label(s5, text='量測位置：自動（全圖最大特徵）\n'
                                               '👆 點影像=選特徵；按住拖曳=拉剖面線'
                                               '（Section）；點剖面圖高度=改門檻',
                                      wraplength=250, foreground='#999', font=('', 8))
            self.lbl_pick.pack(fill='x')

            # ⑥ 儲存
            s6 = step('⑥ 儲存')
            ttk.Button(s6, text='💾 儲存結果（npy + PNG）',
                       command=self._save).pack(fill='x')

            # log
            self.txt = tk.Text(left, height=7, width=38, font=('Consolas', 8))
            self.txt.pack(fill='x', pady=(6, 0))

            # 右側：分頁（頁1 Section 量測＝NanoScope 式；頁2 去卷積詳情）
            self.nb = ttk.Notebook(right)
            self.nb.pack(fill='both', expand=True)
            tab1 = ttk.Frame(self.nb); tab2 = ttk.Frame(self.nb)
            self.nb.add(tab1, text='  Section 量測  ')
            self.nb.add(tab2, text='  去卷積詳情  ')

            # ── 頁1：左大影像 + 右 Section 剖面 + 下方數據表 ──
            self.fig1 = Figure(figsize=(11, 5.2))
            gs = self.fig1.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
            self._ax_sec_img = self.fig1.add_subplot(gs[0])
            self._ax_sec_prof = self.fig1.add_subplot(gs[1])
            self.canvas1 = FigureCanvasTkAgg(self.fig1, master=tab1)
            self.canvas1.get_tk_widget().pack(fill='both', expand=True)
            for ev, fn in (('button_press_event', self._on_press),
                           ('motion_notify_event', self._on_motion),
                           ('button_release_event', self._on_release)):
                self.canvas1.mpl_connect(ev, fn)
            cols = ('curve', 'z', 'xl', 'xr', 'w')
            heads = ('曲線', '量測高度 (nm)', '左緣位置 (nm)',
                     '右緣位置 (nm)', '寬度 (nm)')
            self.tbl = ttk.Treeview(tab1, columns=cols, show='headings', height=2)
            for c, h in zip(cols, heads):
                self.tbl.heading(c, text=h)
                self.tbl.column(c, width=120, anchor='center')
            self.tbl.pack(fill='x', padx=4, pady=(0, 4))

            # ── 頁2：原 2×4 詳情圖 ──
            self.fig = Figure(figsize=(11, 7.2))
            self.canvas = FigureCanvasTkAgg(self.fig, master=tab2)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            self.canvas.mpl_connect('button_press_event', self._on_press)
            self.canvas.mpl_connect('motion_notify_event', self._on_motion)
            self.canvas.mpl_connect('button_release_event', self._on_release)

            # ── 頁3：3D 還原表面（可滑鼠旋轉）──
            tab3 = ttk.Frame(self.nb)
            self.nb.add(tab3, text='  3D 還原  ')
            top3 = ttk.Frame(tab3); top3.pack(fill='x', pady=2)
            self.var_3dsrc = tk.StringVar(value='recon')
            ttk.Radiobutton(top3, text='還原表面', value='recon',
                            variable=self.var_3dsrc,
                            command=self._draw_3d).pack(side='left')
            ttk.Radiobutton(top3, text='輸入影像', value='image',
                            variable=self.var_3dsrc,
                            command=self._draw_3d).pack(side='left')
            ttk.Button(top3, text='⟳ 重繪 3D',
                       command=self._draw_3d).pack(side='left', padx=6)
            ttk.Label(top3, text='  Z 高度比例').pack(side='left')
            self.var_3dz = tk.DoubleVar(value=0.6)
            tk.Scale(top3, from_=0.1, to=3.0, resolution=0.05,
                     orient='horizontal', variable=self.var_3dz, length=150,
                     command=self._apply_3d_z, showvalue=True).pack(side='left')
            ttk.Label(top3, text='（拖曳旋轉；滑桿調 Z 軸高度）',
                      foreground='#888').pack(side='left')
            self.fig3 = Figure(figsize=(8, 6))
            self.ax3 = self.fig3.add_subplot(111, projection='3d')
            self.canvas3 = FigureCanvasTkAgg(self.fig3, master=tab3)
            self.canvas3.get_tk_widget().pack(fill='both', expand=True)
            self._3d_drawn_for = None          # 記錄已繪的 (來源, recon物件id)
            self.nb.bind('<<NotebookTabChanged>>', self._on_tab_change)
            # 探針 R/θ/尺度/樣品變動時，若「自動」開啟則重算視窗半徑
            for v in (self.var_R, self.var_th, self.var_thx, self.var_thy,
                      self.var_px, self.var_sample):
                v.trace_add('write', lambda *_: self.after(60, self._refresh_auto_half))
            # 參數變更 → 自動重跑去卷積（防抖）；半深比例 → 自動重量測
            for v in (self.var_R, self.var_th, self.var_thx, self.var_thy,
                      self.var_px, self.var_half):
                v.trace_add('write', lambda *_: self._schedule_auto_run())
            self.var_wfrac.trace_add('write', lambda *_: self._schedule_measure())
            # 按 Enter 立即套用（不必等防抖）
            for e in (self.e_R, self.e_th, self.e_thx, self.e_thy, self.e_half):
                e.bind('<Return>', lambda _ev: (self._cancel_auto(), self._auto_run()))
            self._refresh()
            self._update_tip_mode()

        def _stat(self, parent, name, row):
            ttk.Label(parent, text=name + '：').grid(row=row, column=0, sticky='w')
            v = ttk.Label(parent, text='—', foreground='#0a7')
            v.grid(row=row, column=1, sticky='w')
            return v

        # ── 探針模式切換（對稱/非對稱、cone/npy 啟停）──────────
        def _update_tip_mode(self):
            cone = self.var_source.get() == 'cone'
            asym = self.var_asym.get()

            def en(w, on): w.config(state=('normal' if on else 'disabled'))
            en(self.e_R, cone); en(self.chk_asym, cone)
            en(self.e_half, cone and not self.var_autohalf.get())   # 自動時鎖住手動欄
            en(self.e_th, cone and not asym)
            en(self.e_thx, cone and asym); en(self.e_thy, cone and asym)
            en(self.btn_tipnpy, not cone)
            self._refresh_auto_half()

        # ── 自動視窗半徑（依載入孔深 + R/θ）───────────────────
        def _auto_half(self):
            if self.image is None:
                return None
            try:
                R = _fnum(self.var_R.get())
                th = (max(_fnum(self.var_thx.get()), _fnum(self.var_thy.get()))
                      if self.var_asym.get() else _fnum(self.var_th.get()))
                px_nm = _fnum(self.var_px.get())
            except (ValueError, tk.TclError):
                return None
            return auto_tip_half(self.image, px_nm, R, th, self.var_sample.get())

        def _refresh_auto_half(self):
            if self.var_source.get() == 'cone' and self.var_autohalf.get():
                h = self._auto_half()
                if h is not None and self.var_half.get() != str(h):
                    self.var_half.set(str(h))

        # ── log（主執行緒安全）────────────────────────────────
        def _log(self, msg):
            self.txt.insert('end', msg + '\n'); self.txt.see('end')

        def _log_safe(self, msg):
            self.after(0, lambda m=msg: self._log(m))

        # ── ① 載入掃描檔（Nanoscope .000 或 .npy）─────────────
        def _load_image(self):
            p = filedialog.askopenfilename(
                title='選擇掃描檔',
                filetypes=[('AFM 掃描 (.000… / .npy)',
                            '*.npy *.000 *.001 *.002 *.003 *.004 '
                            '*.005 *.006 *.007 *.008 *.009'),
                           ('所有檔案', '*.*')])
            if not p:
                return
            try:
                if is_nanoscope_file(p):
                    sample = self.var_sample.get()
                    arr, px_nm, info = load_nanoscope_surface(p, sample)
                    self.var_px.set(f'{px_nm:.2f}')            # 自動帶入尺度
                    src = f'Nanoscope {os.path.basename(p)}\n{info}'
                    self._log(f'解析 Nanoscope：{info}（樣品={sample}）')
                else:
                    arr = np.squeeze(np.load(p)).astype(np.float64)
                    src = (f'{os.path.basename(p)}\nshape={arr.shape}  '
                           f'[{arr.min():.1f}, {arr.max():.1f}] nm')
            except Exception as e:
                messagebox.showerror('讀取失敗', str(e)); return
            if arr.ndim != 2:
                messagebox.showerror('格式錯誤',
                                     f'需要 2D 陣列，讀到 shape={arr.shape}')
                return
            self.image, self.image_path = arr, p
            self.recon = self.certain = self.frac = self.meas = self.meas_in = None
            self.pick = None                     # 新影像重設點選位置
            self.section = None                  # 新影像清除剖面線
            self.lbl_img.config(text=src)
            self._log(f'載入 {os.path.basename(p)} {arr.shape} '
                      f'範圍[{arr.min():.1f}, {arr.max():.1f}]nm')
            self._refresh_auto_half()          # 依新影像孔深自動更新視窗半徑
            if self.var_autohalf.get():
                self._log(f'自動視窗半徑 = {self.var_half.get()} px')
            self._refresh()
            self._schedule_auto_run()          # 載入即自動跑第一輪去卷積

        # ── ③ 載入探針檔 ──────────────────────────────────────
        def _load_tip_npy(self):
            p = filedialog.askopenfilename(
                title='選擇探針 (.npy)',
                filetypes=[('NumPy 陣列', '*.npy'), ('所有檔案', '*.*')])
            if not p:
                return
            self.tip_npy_path = p
            self.lbl_tip.config(text=f'探針檔：{os.path.basename(p)}')
            self._log(f'指定探針檔 {os.path.basename(p)}')

        # ── 依 UI 設定建立探針 ────────────────────────────────
        def _build_tip(self):
            half = int(_fnum(self.var_half.get()))
            px_nm = _fnum(self.var_px.get())
            if self.var_source.get() == 'npy':
                if not self.tip_npy_path:
                    raise ValueError('尚未選擇探針 .npy（或改用廠商 cone）')
                tip = np.squeeze(np.load(self.tip_npy_path)).astype(np.float64)
                tip -= tip.max()
                return tip
            R = _fnum(self.var_R.get())
            if self.var_asym.get():
                tx, ty = _fnum(self.var_thx.get()), _fnum(self.var_thy.get())
                return make_cone_tip_asym(half, px_nm, R, tx, ty)
            th = _fnum(self.var_th.get())
            return make_cone_tip(half, px_nm, R, th)

        # ── 預覽探針 ──────────────────────────────────────────
        def _preview_tip(self):
            try:
                self.tip = self._build_tip()
            except Exception as e:
                messagebox.showerror('探針設定錯誤', str(e)); return
            kind = ('非對稱 cone' if (self.var_source.get() == 'cone'
                    and self.var_asym.get())
                    else '對稱 cone' if self.var_source.get() == 'cone'
                    else '載入探針')
            self._log(f'預覽探針（{kind}）shape={self.tip.shape}')
            self._refresh()

        # ── ④ 執行去卷積（背景執行緒）─────────────────────────
        def _run(self):
            if self.image is None:
                messagebox.showwarning('缺少影像', '請先於 ① 載入影像'); return
            try:
                tip = self._build_tip()
            except Exception as e:
                messagebox.showerror('探針設定錯誤', str(e)); return
            self.tip = tip
            self._busy = True
            self._log('去卷積中…（grey_erosion + certainty）')

            def worker():
                try:
                    res = deconvolve(self.image, tip)
                except Exception as e:
                    def fail():
                        self._busy = False
                        messagebox.showerror('去卷積失敗', str(e))
                    self.after(0, fail)
                    return
                self.after(0, lambda: self._done(res))

            threading.Thread(target=worker, daemon=True).start()

        # ── 參數變更 → 自動重跑（防抖 0.9s，等打完再跑；不完整值略過）──
        def _params_key(self):
            return (self.var_source.get(), self.var_R.get(), self.var_asym.get(),
                    self.var_th.get(), self.var_thx.get(), self.var_thy.get(),
                    self.var_px.get(), self.var_half.get(), self.var_sample.get(),
                    self.tip_npy_path)

        def _cancel_auto(self):
            if self._auto_job:
                self.after_cancel(self._auto_job)
                self._auto_job = None

        def _schedule_auto_run(self):
            if self.image is None:
                return
            self._cancel_auto()
            self._auto_job = self.after(900, self._auto_run)   # 打字停 0.9s 才跑

        def _auto_run(self):
            self._auto_job = None
            if self.image is None:
                return
            if self._busy:                       # 上一輪還在跑 → 稍後重試
                self._auto_job = self.after(300, self._auto_run)
                return
            key = self._params_key()
            if key == self._last_run_key and self.recon is not None:
                return                           # 參數沒實質改變 → 不重跑（免閃爍）
            try:
                tip = self._build_tip()
            except Exception:
                return                           # 欄位輸入中/不完整 → 不跳錯誤視窗
            self.tip = tip
            self._last_run_key = key
            self._busy = True
            self._log('參數變更 → 自動重新去卷積…')

            def worker():
                try:
                    res = deconvolve(self.image, tip)
                except Exception as e:
                    self.after(0, lambda: (self._log(f'自動重跑失敗：{e}'),
                                           setattr(self, '_busy', False)))
                    return
                self.after(0, lambda: self._done(res))

            threading.Thread(target=worker, daemon=True).start()

        def _schedule_measure(self):
            if self._meas_job:
                self.after_cancel(self._meas_job)
            self._meas_job = self.after(500, self._measure_if_ready)

        def _measure_if_ready(self):
            self._meas_job = None
            if self.recon is not None:
                self._measure()

        # ── 樣品類型切換：Nanoscope 檔需以新基線方向重新載入 ──
        def _on_sample_change(self):
            if self.image_path and is_nanoscope_file(self.image_path):
                sample = self.var_sample.get()
                try:
                    arr, px_nm, info = load_nanoscope_surface(self.image_path, sample)
                except Exception as e:
                    self._log(f'重新載入失敗：{e}'); return
                self.image = arr
                self.var_px.set(f'{px_nm:.2f}')
                self.recon = self.certain = self.frac = None
                self.meas = self.meas_in = None
                self.pick = None
                self.lbl_img.config(
                    text=f'Nanoscope {os.path.basename(self.image_path)}\n{info}')
                self._log(f'樣品類型 → {sample}：已重新載入並重新去傾斜')
                self._refresh()
            self._refresh_auto_half()
            self._schedule_auto_run()

        def _done(self, res):
            self._busy = False
            self.recon, self.certain = res['recon'], res['certain']
            self.frac = res['certain_frac']
            self.lbl_cert.config(text=f'{self.frac*100:.1f} %')
            self.lbl_din.config(text=f'{depth_of(self.image):.1f} nm')
            self.lbl_dre.config(text=f'{depth_of(self.recon):.1f} nm')
            # 下界性質檢查（誠實回報）
            ok = bool(np.all(self.recon <= self.image + 1e-6))
            self._log(f'完成：certain={self.frac*100:.1f}%  '
                      f'還原起伏={depth_of(self.recon):.1f}nm  下界 s_r≤i={ok}')
            self._log(f'  （紅色死角=探針碰不到、不可信；引擎只給確定下界不腦補）')
            self._measure()                    # 去卷積後自動量測寬度並繪圖

        # ── 滑鼠互動總管 ──────────────────────────────────────
        #   頁1 掃描影像：按住拖曳＝拉剖面線；拖端點＝調整
        #   頁1 Section 剖面：點/拖＝移動較近的游標
        #   頁2 影像/還原：點＝選特徵；頁2 特徵剖面：點＝改門檻
        def _on_press(self, event):
            if event.inaxes is None or event.xdata is None:
                return
            ax = event.inaxes
            if ax is getattr(self, '_ax_sec_img', None):     # 頁1 掃描影像
                if self.image is None:
                    return
                if self.section:                             # 命中端點 → 拖曳調整
                    tol = max(6, 0.02 * self.image.shape[1])
                    for key in ('p0', 'p1'):
                        py_, px_ = self.section[key]
                        if abs(event.xdata - px_) < tol and abs(event.ydata - py_) < tol:
                            self._drag = ('endpoint', key)
                            return
                self._drag = ('newline', (float(event.ydata), float(event.xdata)),
                              (event.x, event.y))
            elif self.section and ax is getattr(self, '_ax_sec_prof', None):
                if event.xdata is None:
                    return
                self.section['xpos'] = float(event.xdata)    # 頁1 剖面 → 拖位置游標
                self._drag = ('xcur',)
                self._update_section_artists()
            elif ax in (getattr(self, '_ax_img', None), getattr(self, '_ax_rec', None)):
                self._drag = ('pick', (float(event.ydata), float(event.xdata)))
            elif ax in (getattr(self, '_ax_profx', None),
                        getattr(self, '_ax_profy', None)):
                self._threshold_click(event)

        def _on_motion(self, event):
            if not self._drag:
                return
            kind = self._drag[0]
            if kind == 'newline':
                x0, y0 = self._drag[2]                        # 移動 >4 螢幕px 才建線
                if abs(event.x - x0) <= 4 and abs(event.y - y0) <= 4:
                    return
                if event.inaxes is not self._ax_sec_img or event.xdata is None:
                    return
                self.section = {'p0': self._drag[1],
                                'p1': (float(event.ydata), float(event.xdata)),
                                'xpos': None}
                self.section['xpos'] = self._default_xpos()  # 預設抓半高左緣
                self._drag = ('endpoint', 'p1')
                self._refresh_section()                      # 建立 artists
            elif kind == 'endpoint':
                if event.inaxes is not self._ax_sec_img or event.xdata is None:
                    return
                H, W = self.image.shape
                self.section[self._drag[1]] = (
                    float(np.clip(event.ydata, 0, H - 1)),
                    float(np.clip(event.xdata, 0, W - 1)))
                self._update_section_artists()
            elif kind == 'xcur':
                if event.inaxes is not self._ax_sec_prof or event.xdata is None:
                    return
                self.section['xpos'] = float(event.xdata)
                self._update_section_artists()

        def _on_release(self, event):
            drag, self._drag = self._drag, None
            if not drag:
                return
            if drag[0] == 'pick':                            # 頁2：點選特徵量測
                y, x = drag[1]
                self._pick_feature(y, x)
            elif drag[0] in ('endpoint', 'xcur') and self.section:
                L = self._section_lines()
                self._log(f"Section：{L['h']}  {L['i']}  {L['r']}  {L['d']}".strip())
                self._refresh_section()                      # 收尾統一重繪

        # ── 點特徵（原本的影像點選量測）──────────────────────
        def _pick_feature(self, y, x):
            if self.recon is None:
                return
            self.pick = (float(y), float(x))
            self.lbl_pick.config(
                text=f'量測位置：點選 (x={int(x)}, y={int(y)})'
                     f'　→ 按「↺ 自動」可還原')
            self._log(f'點選量測位置 (x={int(x)}, y={int(y)})')
            self._measure()

        # ── 點剖面圖高度＝設定量測門檻 ────────────────────────
        def _threshold_click(self, event):
            if self.recon is None or not self.meas_in or event.ydata is None:
                return
            z = float(event.ydata)
            if self.var_sample.get() == 'bump':
                base = float(np.percentile(self.image, 10))
                frac = (z - base) / self.meas_in['amp_nm']
            else:
                base = float(np.percentile(self.image, 95))
                frac = (base - z) / self.meas_in['amp_nm']
            frac = min(0.95, max(0.05, frac))
            self.var_wfrac.set(f'{frac:.2f}')
            self._log(f'剖面點選高度 {z:.2f}nm → 門檻比例 {frac:.2f}，重新量測')
            self._measure()

        # ── Section 剖面線：取樣 / 座標換算 / 讀值 / 清除 ────
        def _section_profile(self, surf, n=240):
            """沿剖面線取樣：回傳 (距離nm 陣列, 高度nm 陣列)；線長為 0 回 None。"""
            (y0, x0), (y1, x1) = self.section['p0'], self.section['p1']
            dist_px = float(np.hypot(y1 - y0, x1 - x0))
            if dist_px < 1e-6:
                return None
            ts = np.linspace(0.0, 1.0, n)
            z = map_coordinates(surf, [y0 + (y1 - y0) * ts, x0 + (x1 - x0) * ts],
                                order=1, mode='nearest')
            try:
                px = _fnum(self.var_px.get())
            except ValueError:
                px = 1.0
            return ts * dist_px * px, z

        def _section_pt(self, t):
            (y0, x0), (y1, x1) = self.section['p0'], self.section['p1']
            return y0 + (y1 - y0) * t, x0 + (x1 - x0) * t

        def _default_xpos(self):
            """建線後的預設游標位置：影像曲線半高的左緣（落在特徵上升坡）。"""
            prof = self._section_profile(self.image)
            if prof is None:
                return 0.0
            xs, z = prof
            if self.var_sample.get() == 'bump':
                base = float(np.percentile(z, 20))
                zlev = base + 0.5 * (float(z.max()) - base)
            else:
                base = float(np.percentile(z, 80))
                zlev = base - 0.5 * (base - float(z.min()))
            c = self._level_cross(self.image, zlev)
            return c['xl'] if c else float(xs[-1]) * 0.25

        def _curve_chord(self, surf, xpos):
            """B 型等高：取『曲線在游標位置 xpos 的高度』為該曲線自己的量測高度，
            回傳 (zlev, {'xl','xr','w'})——同一條曲線的左右緣必等高。"""
            prof = self._section_profile(surf)
            if prof is None or xpos is None:
                return None, None
            xs, z = prof
            zlev = float(np.interp(np.clip(xpos, xs[0], xs[-1]), xs, z))
            return zlev, self._level_cross(surf, zlev)

        def _level_cross(self, surf, zlev):
            """等高量測：在剖面上找『主特徵通過高度 zlev 的左右緣』（亞像素內插）。

            從峰頂（凸起）/谷底（孔洞）向兩側走到跨越 zlev 處——影像與還原用
            **同一高度**量，兩者寬度直接相減即為去卷積修正量。
            高度不落在特徵範圍內回 None。
            """
            prof = self._section_profile(surf)
            if prof is None or zlev is None:
                return None
            xs, z = prof
            n = len(z)
            if self.var_sample.get() == 'bump':
                ipk = int(np.argmax(z)); inside = z >= zlev
            else:
                ipk = int(np.argmin(z)); inside = z <= zlev
            if not inside[ipk]:
                return None
            L = ipk
            while L > 0 and inside[L - 1]:
                L -= 1
            R = ipk
            while R < n - 1 and inside[R + 1]:
                R += 1

            def interp(a, b):                    # a=特徵內緣、b=外側鄰點
                if z[b] == z[a]:
                    return float(xs[a])
                f = (zlev - z[a]) / (z[b] - z[a])
                return float(xs[a] + f * (xs[b] - xs[a]))

            xl = interp(L, L - 1) if L > 0 else float(xs[0])
            xr = interp(R, R + 1) if R < n - 1 else float(xs[-1])
            return {'xl': xl, 'xr': xr, 'w': xr - xl}

        def _section_clear(self):
            if self.section is None:
                return
            self.section = None
            self._log('已清除剖面線')
            self._refresh()

        def _pick_reset(self):
            self.pick = None
            self.lbl_pick.config(text='量測位置：自動（全圖最大特徵）\n'
                                      '👆 點影像/還原圖=選特徵；'
                                      '點剖面圖高度=改量測門檻')
            if self.recon is not None:
                self._measure()

        # ── ⑤ 量測還原表面特徵寬度（FWHM；自動或點選位置）────
        def _measure(self):
            if self.recon is None:
                messagebox.showwarning('尚無結果', '請先執行 ④ 去卷積'); return
            try:
                frac = min(0.95, max(0.05, _fnum(self.var_wfrac.get())))
            except ValueError:
                frac = 0.5
            px_nm = _fnum(self.var_px.get())
            sample = self.var_sample.get()
            self.meas = measure_feature_width(self.recon, px_nm, sample, frac,
                                              at=self.pick)
            m_in = measure_feature_width(self.image, px_nm, sample, frac,
                                         at=self.pick)
            self.meas_in = m_in
            if self.meas:
                self.lbl_wre.config(
                    text=f"{self.meas['width_x_nm']:.1f}×{self.meas['width_y_nm']:.1f}"
                         f" nm ⌀{self.meas['equiv_diam_nm']:.1f}")
            else:
                self.lbl_wre.config(text='（無明顯特徵）')
            if m_in:
                self.lbl_win.config(text=f"{m_in['equiv_diam_nm']:.1f} nm")
            if self.meas and m_in:
                self._log(f"寬度@{int(frac*100)}%：還原 "
                          f"{self.meas['width_x_nm']:.1f}×{self.meas['width_y_nm']:.1f}nm "
                          f"⌀{self.meas['equiv_diam_nm']:.1f}；"
                          f"影像 ⌀{m_in['equiv_diam_nm']:.1f}nm")
            self._refresh()

        # ── ⑥ 儲存 ───────────────────────────────────────────
        def _save(self):
            if self.recon is None:
                messagebox.showwarning('尚無結果', '請先執行 ④ 去卷積'); return
            d = filedialog.askdirectory(title='選擇輸出資料夾')
            if not d:
                return
            stem = os.path.splitext(os.path.basename(
                self.image_path or 'scan'))[0]
            np.save(os.path.join(d, f'{stem}_reconstructed.npy'), self.recon)
            np.save(os.path.join(d, f'{stem}_certain.npy'), self.certain)
            png = os.path.join(d, f'{stem}_blind_deconv.png')
            save_panels(self.image, self.recon, self.certain, self.tip, self.frac,
                        png, title=f'Certainty deconvolution — {stem}',
                        meas=self.meas, meas_in=self.meas_in)
            self._log(f'已儲存至 {d}')
            messagebox.showinfo('完成', f'結果已存至：\n{d}')

        # ── 統一重繪：頁1 Section + 頁2 詳情 ──────────────────
        def _refresh(self):
            self._refresh_section()
            self._refresh_detail()

        # ── 頁1：掃描影像 + Section 剖面（NanoScope 式）────────
        def _refresh_section(self):
            axi, axp = self._ax_sec_img, self._ax_sec_prof
            axi.clear(); axp.clear()
            axi.set_xticks([]); axi.set_yticks([])
            if self.image is None:
                axi.set_title('掃描影像（尚未載入）', fontsize=9)
                axp.set_xticks([]); axp.set_yticks([])
                axp.set_title('Section 剖面（載入後在左圖拖線量測）', fontsize=9)
                self._clear_table(); self._sa = {}
                self.fig1.tight_layout(); self.canvas1.draw()
                return
            axi.imshow(self.image, cmap='afmhot')            # AFM 金銅色調
            axi.set_title('掃描影像　（按住拖曳＝拉剖面線；拖白點＝調整）', fontsize=9)
            if self.section:
                self._draw_section(axi, axp)
                self._update_table()
            else:
                axp.set_xticks([]); axp.set_yticks([])
                axp.set_title('Section 剖面\n（在左圖按住滑鼠拖出一條線）', fontsize=9)
                self._clear_table(); self._sa = {}
            self.fig1.tight_layout(); self.canvas1.draw()

        # ── 頁2：去卷積詳情（原 2×4 圖）───────────────────────
        def _refresh_detail(self):
            self.fig.clf()
            axs = self.fig.subplots(2, 4)
            self._ax_img, self._ax_rec = axs[0, 0], axs[0, 1]   # 供點選量測判定
            # 影像類子圖關掉刻度；剖面折線圖保留刻度（於下方另設）
            for ax in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]):
                ax.set_xticks([]); ax.set_yticks([])

            # 共用色階（影像+還原）
            imgs = [a for a in (self.image, self.recon) if a is not None]
            if imgs:
                vmin = min(a.min() for a in imgs)
                vmax = max(a.max() for a in imgs)
            else:
                vmin = vmax = None

            if self.image is not None:
                axs[0, 0].imshow(self.image, cmap='viridis', vmin=vmin, vmax=vmax)
                axs[0, 0].set_title(f'輸入影像\nmin={self.image.min():.1f} '
                                    f'max={self.image.max():.1f} nm', fontsize=9)
                if self.meas_in:                       # 影像寬度（橘色，對照）
                    _draw_width(axs[0, 0], self.meas_in, color='#ff8c00')
            else:
                axs[0, 0].set_title('輸入影像（尚未載入）', fontsize=9)

            if self.recon is not None:
                axs[0, 1].imshow(self.recon, cmap='viridis', vmin=vmin, vmax=vmax)
                axs[0, 1].set_title(f'還原表面（erosion）\n'
                                    f'min={self.recon.min():.1f} nm 確定下界',
                                    fontsize=9)
                if self.meas:                          # 疊上寬度量測線與數值
                    _draw_width(axs[0, 1], self.meas)
                axs[0, 2].imshow(self.recon, cmap='gray', vmin=vmin, vmax=vmax)
                ov = np.zeros((*self.certain.shape, 4))
                ov[~self.certain] = [1, 0, 0, 0.40]
                axs[0, 2].imshow(ov)
                axs[0, 2].set_title(f'certainty map\ncertain='
                                    f'{self.frac*100:.1f}% (紅=碰不到)', fontsize=9)
            else:
                axs[0, 1].set_title('還原表面（待執行）', fontsize=9)
                axs[0, 2].set_title('certainty map（待執行）', fontsize=9)

            # 探針視覺化
            if self.tip is not None:
                t = self.tip
                h = t.shape[0] // 2
                im = axs[1, 0].imshow(t, cmap='magma')
                self.fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
                axs[1, 0].set_title('探針 2D（apex=0）', fontsize=9)
                xs = np.arange(-h, t.shape[1] - h)
                axs[1, 1].plot(xs, t[h, :], color='#1d9e75', lw=2)
                axs[1, 1].axhline(0, color='gray', lw=0.6)
                axs[1, 1].set_title('探針 X 剖面', fontsize=9)
                axs[1, 1].grid(alpha=0.3); axs[1, 1].set_xlabel('px')
                ys = np.arange(-h, t.shape[0] - h)
                axs[1, 2].plot(ys, t[:, h], color='#d1495b', lw=2)
                axs[1, 2].axhline(0, color='gray', lw=0.6)
                axs[1, 2].set_title('探針 Y 剖面', fontsize=9)
                axs[1, 2].grid(alpha=0.3); axs[1, 2].set_xlabel('px')
            else:
                for j, name in enumerate(('探針 2D', '探針 X 剖面', '探針 Y 剖面')):
                    axs[1, j].set_xticks([]); axs[1, j].set_yticks([])
                    axs[1, j].set_title(f'{name}（未設定）', fontsize=9)

            # 右欄：特徵剖面 X / Y（含門檻量測線）
            self._ax_profx, self._ax_profy = axs[0, 3], axs[1, 3]
            self._draw_feature_profile(axs[0, 3], horiz=True)
            self._draw_feature_profile(axs[1, 3], horiz=False)

            self.fig.tight_layout()
            self.canvas.draw()

        # ── NanoScope 式 Pair 表：影像/還原各一列（游標間量測）──
        def _clear_table(self):
            for r in self.tbl.get_children():
                self.tbl.delete(r)

        def _update_table(self):
            self._clear_table()
            if not self.section:
                return
            xpos = self.section.get('xpos')
            for surf, lab, tag in ((self.image, '影像', 'img'),
                                   (self.recon, '還原', 'rec')):
                if surf is None:
                    continue
                zlev, c = self._curve_chord(surf, xpos)
                if c:
                    vals = (lab, f'{zlev:.2f}', f"{c['xl']:.2f}",
                            f"{c['xr']:.2f}", f"{c['w']:.2f}")
                else:
                    vals = (lab, '—' if zlev is None else f'{zlev:.2f}',
                            '—', '—', '（游標處無交點）')
                self.tbl.insert('', 'end', tags=(tag,), values=vals)
            self.tbl.tag_configure('img', foreground='#d2691e')
            self.tbl.tag_configure('rec', foreground='#c0392b')

        # ── 頁3：3D 表面（切到該頁才繪，避免拖慢其他頁）────────
        def _on_tab_change(self, _evt=None):
            if self.nb.index('current') == 2:            # 第3頁 = 3D
                key = (self.var_3dsrc.get(),
                       id(self.recon) if self.var_3dsrc.get() == 'recon'
                       else id(self.image))
                if key != self._3d_drawn_for:            # 資料沒變就不重畫
                    self._draw_3d()

        def _draw_3d(self):
            surf = self.recon if self.var_3dsrc.get() == 'recon' else self.image
            self.ax3.clear()
            if surf is None:
                self.ax3.set_title('（尚無資料——請先載入並去卷積）', fontsize=10)
                self.canvas3.draw(); return
            H, W = surf.shape
            step = max(1, max(H, W) // 90)               # 降取樣到 ~90 供旋轉流暢
            z = np.asarray(surf[::step, ::step], dtype=float)
            try:
                px = _fnum(self.var_px.get())
            except ValueError:
                px = 1.0
            xs = np.arange(z.shape[1]) * step * px
            ys = np.arange(z.shape[0]) * step * px
            X, Y = np.meshgrid(xs, ys)
            self.ax3.plot_surface(X, Y, z, cmap='afmhot', linewidth=0,
                                  antialiased=False,
                                  rcount=z.shape[0], ccount=z.shape[1])
            lab = '還原表面' if self.var_3dsrc.get() == 'recon' else '輸入影像'
            self.ax3.set_title(f'{lab} 3D（{z.shape[1]}×{z.shape[0]} 取樣；拖曳旋轉）',
                               fontsize=10)
            self.ax3.set_xlabel('X (nm)'); self.ax3.set_ylabel('Y (nm)')
            self.ax3.set_zlabel('Z (nm)')
            try:
                self.ax3.set_box_aspect((1, 1, float(self.var_3dz.get())))
            except Exception:
                pass
            self.canvas3.draw()
            self._3d_drawn_for = (self.var_3dsrc.get(),
                                  id(self.recon) if self.var_3dsrc.get() == 'recon'
                                  else id(self.image))

        def _apply_3d_z(self, *_):
            """只調 Z 軸高度比例（不重算表面），保持滑桿拖曳流暢。"""
            if getattr(self, 'ax3', None) is None or self._3d_drawn_for is None:
                return
            try:
                self.ax3.set_box_aspect((1, 1, float(self.var_3dz.get())))
                self.canvas3.draw_idle()
            except Exception:
                pass

        # ── 特徵剖面圖：影像(橘) vs 還原(紅)，量測跨距畫在門檻高度 ──
        def _draw_feature_profile(self, ax, horiz=True):
            name = '特徵剖面 X（沿量測線）' if horiz else '特徵剖面 Y（沿量測線）'
            if not (self.meas or self.meas_in):
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{name}\n（量測後顯示）', fontsize=9)
                return
            try:
                px = _fnum(self.var_px.get())
            except ValueError:
                px = 1.0
            sample = self.var_sample.get()
            for surf, m, color, lab in ((self.image, self.meas_in, '#ff8c00', '影像'),
                                        (self.recon, self.meas, '#d1495b', '還原')):
                if surf is None or not m:
                    continue
                if horiz:
                    line = surf[m['cy'], :]; lo, hi = m['x0'], m['x1']
                else:
                    line = surf[:, m['cx']]; lo, hi = m['y0'], m['y1']
                ax.plot(np.arange(len(line)) * px, line, color=color, lw=1.4, label=lab)
                # 量測跨距畫在『該量測自己的門檻高度』（frac×振幅，相對基線）
                if sample == 'bump':
                    thr_z = np.percentile(surf, 10) + m['frac'] * m['amp_nm']
                else:
                    thr_z = np.percentile(surf, 95) - m['frac'] * m['amp_nm']
                ax.plot([lo * px, hi * px], [thr_z, thr_z],
                        color=color, lw=2.2, ls='--')
                # 影像標在線上方、還原標在線下方，避免兩標籤重疊
                above = (lab == '影像') if sample == 'bump' else (lab != '影像')
                ax.annotate(f'{lab} {(hi - lo) * px:.1f}nm',
                            ((lo + hi) / 2 * px, thr_z), fontsize=8, color=color,
                            ha='center', va='bottom' if above else 'top',
                            xytext=(0, 3 if above else -3),
                            textcoords='offset points')
            # 視野鎖定特徵附近（±2.5× 跨距），才看得清楚曲線形狀
            m = self.meas or self.meas_in
            lo, hi = (m['x0'], m['x1']) if horiz else (m['y0'], m['y1'])
            c, w = (lo + hi) / 2, max(hi - lo, 4)
            ax.set_xlim(max(0.0, (c - 2.5 * w) * px), (c + 2.5 * w) * px)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel('nm'); ax.set_ylabel('nm')
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')

        # ── Section 繪製（B 型等高：各曲線在游標處自己的高度取水平弦）──
        def _section_lines(self):
            """量測文字：h=游標位置、i=影像弦、r=還原弦、d=寬度差。"""
            xpos = self.section.get('xpos') if self.section else None
            if xpos is None:
                return {'h': '', 'i': '', 'r': '', 'd': ''}
            out = {'h': f'游標位置 {xpos:.1f} nm', 'i': '', 'r': '', 'd': ''}
            zi, ci = self._curve_chord(self.image, xpos)
            out['i'] = (f"影像 寬 {ci['w']:.2f} nm（高 {zi:.2f}）" if ci
                        else '影像：游標處無交點')
            cr = None
            if self.recon is not None:
                zr, cr = self._curve_chord(self.recon, xpos)
                out['r'] = (f"還原 寬 {cr['w']:.2f} nm（高 {zr:.2f}）" if cr
                            else '還原：游標處無交點')
            if ci and cr:
                out['d'] = f"寬度差 {ci['w'] - cr['w']:+.2f} nm（探針撐寬量）"
            return out

        def _draw_section(self, axi, axp):
            sa = {}
            (y0, x0), (y1, x1) = self.section['p0'], self.section['p1']
            xpos = self.section.get('xpos')
            # 掃描影像上：白線 + 端點方塊 + 兩曲線交點（橘=影像、紅=還原）
            sa['line'], = axi.plot([x0, x1], [y0, y1], '-', color='white',
                                   lw=1.6, alpha=0.95)
            sa['ends'], = axi.plot([x0, x1], [y0, y1], 's', color='white',
                                   ms=7, mec='black')
            sa['mi'], = axi.plot([], [], 'o', color='#ff8c00', ms=7, mec='white')
            sa['mr'], = axi.plot([], [], 'o', color='#d1495b', ms=7, mec='white')
            prof_i = self._section_profile(self.image)
            if prof_i is not None:
                xs, zi = prof_i
                sa['prof_img'], = axp.plot(xs, zi, color='#ff8c00', lw=1.5,
                                           label='影像')
                if self.recon is not None:
                    _, zr = self._section_profile(self.recon)
                    sa['prof_rec'], = axp.plot(xs, zr, color='#d1495b', lw=1.5,
                                               label='還原')
                # 位置游標（灰垂線）＋各曲線在自己高度的水平弦
                sa['vcur'] = axp.axvline(xpos, color='#555', ls='--', lw=1.3,
                                         alpha=0.8)
                sa['di'], = axp.plot([], [], 'o', color='#ff8c00', ms=7,
                                     mec='k', zorder=5)
                sa['hli'], = axp.plot([], [], ':', color='#ff8c00', lw=1.6)
                sa['dr'], = axp.plot([], [], 'o', color='#d1495b', ms=7,
                                     mec='k', zorder=5)
                sa['hlr'], = axp.plot([], [], ':', color='#d1495b', lw=1.6)
                # 數字標註：色碼對應曲線，字放大加粗
                tb = dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.92)
                sa['stxt_h'] = axp.text(0.02, 0.98, '', transform=axp.transAxes,
                                        va='top', fontsize=12, fontweight='bold',
                                        color='#222', bbox=tb)
                sa['stxt_i'] = axp.text(0.02, 0.885, '', transform=axp.transAxes,
                                        va='top', fontsize=12, fontweight='bold',
                                        color='#e8820e', bbox=tb)
                sa['stxt_r'] = axp.text(0.02, 0.79, '', transform=axp.transAxes,
                                        va='top', fontsize=12, fontweight='bold',
                                        color='#c0392b', bbox=tb)
                sa['stxt_d'] = axp.text(0.02, 0.695, '', transform=axp.transAxes,
                                        va='top', fontsize=11, fontweight='bold',
                                        color='#1a7a4a', bbox=tb)
                axp.set_title('Section 剖面（點/拖 ＝ 移動位置游標，'
                              '各曲線在自己高度取水平弦）', fontsize=9)
                axp.set_xlabel('沿線距離 nm'); axp.set_ylabel('高度 nm')
                axp.grid(alpha=0.3); axp.legend(fontsize=8, loc='upper right')
            self._sa = sa
            self._update_section_artists(light=False)

        def _update_section_artists(self, light=True):
            """更新 B 型等高 artists + 表格；light=True 時只 draw_idle（拖曳滑順）。"""
            sa = self._sa
            if not sa or self.section is None:
                self._refresh_section(); return
            (y0, x0), (y1, x1) = self.section['p0'], self.section['p1']
            xpos = self.section.get('xpos')
            if 'line' in sa:
                sa['line'].set_data([x0, x1], [y0, y1])
                sa['ends'].set_data([x0, x1], [y0, y1])
            prof_i = self._section_profile(self.image)
            if prof_i is not None and 'prof_img' in sa:
                xs, zi = prof_i
                total = xs[-1]
                xpos = float(np.clip(xpos, 0.0, total))
                self.section['xpos'] = xpos
                sa['prof_img'].set_data(xs, zi)
                if 'prof_rec' in sa and self.recon is not None:
                    _, zr = self._section_profile(self.recon)
                    sa['prof_rec'].set_data(xs, zr)
                sa['vcur'].set_xdata([xpos, xpos])
                # 各曲線：游標處自己的高度 → 水平弦（兩端點必同高）；映射回白線
                z_i, ci = self._curve_chord(self.image, xpos)
                if ci:
                    sa['di'].set_data([ci['xl'], ci['xr']], [z_i, z_i])
                    sa['hli'].set_data([ci['xl'], ci['xr']], [z_i, z_i])
                    ts = [ci['xl'] / total, ci['xr'] / total]
                    pts = [self._section_pt(t) for t in ts]
                    sa['mi'].set_data([p[1] for p in pts], [p[0] for p in pts])
                else:
                    sa['di'].set_data([], []); sa['hli'].set_data([], [])
                    sa['mi'].set_data([], [])
                z_r, cr = (self._curve_chord(self.recon, xpos)
                           if self.recon is not None else (None, None))
                if cr:
                    sa['dr'].set_data([cr['xl'], cr['xr']], [z_r, z_r])
                    sa['hlr'].set_data([cr['xl'], cr['xr']], [z_r, z_r])
                    ts = [cr['xl'] / total, cr['xr'] / total]
                    pts = [self._section_pt(t) for t in ts]
                    sa['mr'].set_data([p[1] for p in pts], [p[0] for p in pts])
                else:
                    sa['dr'].set_data([], []); sa['hlr'].set_data([], [])
                    sa['mr'].set_data([], [])
                L = self._section_lines()
                if 'stxt_h' in sa:
                    sa['stxt_h'].set_text(L['h'])
                    sa['stxt_i'].set_text(L['i'])
                    sa['stxt_r'].set_text(L['r'])
                    sa['stxt_d'].set_text(L['d'])
                ax = sa['prof_img'].axes
                ax.relim(); ax.autoscale_view()
            self._update_table()
            if light:
                self.canvas1.draw_idle()

    App().mainloop()


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='通用 AFM 探針去卷積（Villarrubia 形態學 certainty，零訓練）')
    ap.add_argument('image', nargs='?',
                    help='輸入掃描：Nanoscope .000 原始檔 或 .npy（nm 單位 2D 陣列）')
    ap.add_argument('--sample', choices=['hole', 'bump'], default='hole',
                    help='樣品類型（影響 .000 基線方向；預設 hole 孔洞）')
    ap.add_argument('--tip', help='已知探針 .npy（apex 置中=0，向外為負）')
    ap.add_argument('--cone-R', type=float, help='改用廠商 cone：尖端球半徑 R (nm)')
    ap.add_argument('--cone-theta', type=float, default=25.0, help='cone 半錐角 θ (°)')
    ap.add_argument('--tip-half', default='auto',
                    help='cone 視窗半徑 px；或 auto（依載入孔深+R/θ 自動換算，預設）')
    ap.add_argument('--px-nm', type=float, default=39.1, help='影像每 px nm（預設 39.1）')
    ap.add_argument('--out', default='blind_out', help='輸出資料夾')
    ap.add_argument('--demo', action='store_true', help='跑合成資料驗證 + 教學圖')
    ap.add_argument('--gui', action='store_true', help='啟動圖形介面（步驟式操作）')
    args = ap.parse_args()

    # 無參數 → 預設啟動 GUI；或明確指定 --gui
    if args.gui or (args.image is None and not args.demo):
        launch_gui()
        return
    if args.demo:
        run_demo('blind_demo')
        return

    os.makedirs(args.out, exist_ok=True)
    if is_nanoscope_file(args.image):
        image, px_nm, info = load_nanoscope_surface(args.image, args.sample)
        args.px_nm = px_nm                       # 以檔頭尺度覆寫，確保探針/影像一致
        print(f"輸入 Nanoscope：{args.image}  {info}  (樣品={args.sample})")
    else:
        image = np.squeeze(np.load(args.image)).astype(np.float64)
        print(f"輸入影像：{args.image}  shape={image.shape}  "
              f"範圍[{image.min():.1f}, {image.max():.1f}] nm")

    if args.tip:
        tip = np.load(args.tip).astype(np.float64); tip -= tip.max()
        print(f"探針：載入 {args.tip}  shape={tip.shape}")
    elif args.cone_R is not None:
        if str(args.tip_half).lower() == 'auto':
            half = auto_tip_half(image, args.px_nm, args.cone_R,
                                 args.cone_theta, args.sample)
            print(f"探針視窗半徑：auto → {half}px "
                  f"（依 {args.sample} 起伏 + R={args.cone_R}nm θ={args.cone_theta}°）")
        else:
            half = int(args.tip_half)
        tip = make_cone_tip(half, args.px_nm, args.cone_R, args.cone_theta)
        print(f"探針：廠商 cone R={args.cone_R}nm θ={args.cone_theta}° "
              f"({tip.shape[0]}px @ {args.px_nm}nm/px)")
    else:
        print("錯誤：需指定 --tip <檔> 或 --cone-R <nm>（探針來源）")
        print("  盲估探針請用 Gwyddion 的 itip_estimate 跑校正光柵，匯出後 --tip 餵入。")
        return

    res = deconvolve(image, tip)
    recon = res['recon']
    print(f"\n去卷積結果：")
    print(f"  影像起伏 = {depth_of(image):.1f} nm")
    print(f"  還原起伏 = {depth_of(recon):.1f} nm  (確定下界，不會過度去卷積)")
    print(f"  可信像素 certain = {res['certain_frac']*100:.1f}%"
          f"（其餘為探針碰不到的死角，僅下界）")

    # 特徵寬度量測（還原 vs 影像對照）
    meas = measure_feature_width(recon, args.px_nm, args.sample)
    m_in = measure_feature_width(image, args.px_nm, args.sample)
    if meas and m_in:
        print(f"  還原寬度@50% = {meas['width_x_nm']:.1f}×{meas['width_y_nm']:.1f} nm "
              f"(⌀{meas['equiv_diam_nm']:.1f})  ← 影像 ⌀{m_in['equiv_diam_nm']:.1f} nm")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.out, f'{stem}_reconstructed.npy'), recon)
    np.save(os.path.join(args.out, f'{stem}_certain.npy'), res['certain'])
    save_panels(image, recon, res['certain'], tip, res['certain_frac'],
                os.path.join(args.out, f'{stem}_blind_deconv.png'),
                title=f'Certainty deconvolution — {stem}', meas=meas, meas_in=m_in)
    print(f"\n完成。結果存於：{args.out}/")


if __name__ == '__main__':
    main()
