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
from scipy.ndimage import grey_dilation, grey_erosion, median_filter


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
def save_panels(image, recon, certain, tip, certain_frac, out_path, title=''):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    vmin = min(image.min(), recon.min())
    vmax = max(image.max(), recon.max())

    im0 = ax[0].imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title(f'Input image (AFM)\nmin={image.min():.1f} max={image.max():.1f} nm')
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title(f'Reconstructed surface (erosion)\n'
                    f'min={recon.min():.1f} (certain lower bound)')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

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
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import threading
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

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
                            variable=self.var_sample).pack(side='left')
            ttk.Radiobutton(srow, text='凸起', value='bump',
                            variable=self.var_sample).pack(side='left')
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
            self.lbl_din = self._stat(g, '影像孔深', 1)
            self.lbl_dre = self._stat(g, '還原孔深（下界）', 2)

            # ⑥ 儲存
            s6 = step('⑥ 儲存')
            ttk.Button(s6, text='💾 儲存結果（npy + PNG）',
                       command=self._save).pack(fill='x')

            # log
            self.txt = tk.Text(left, height=7, width=38, font=('Consolas', 8))
            self.txt.pack(fill='x', pady=(6, 0))

            # 右側圖
            self.fig = Figure(figsize=(8.5, 7.4))
            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            # 探針 R/θ/尺度/樣品變動時，若「自動」開啟則重算視窗半徑
            for v in (self.var_R, self.var_th, self.var_thx, self.var_thy,
                      self.var_px, self.var_sample):
                v.trace_add('write', lambda *_: self.after(60, self._refresh_auto_half))
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
                R = float(self.var_R.get())
                th = (max(float(self.var_thx.get()), float(self.var_thy.get()))
                      if self.var_asym.get() else float(self.var_th.get()))
                px_nm = float(self.var_px.get())
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
            self.recon = self.certain = self.frac = None
            self.lbl_img.config(text=src)
            self._log(f'載入 {os.path.basename(p)} {arr.shape} '
                      f'範圍[{arr.min():.1f}, {arr.max():.1f}]nm')
            self._refresh_auto_half()          # 依新影像孔深自動更新視窗半徑
            if self.var_autohalf.get():
                self._log(f'自動視窗半徑 = {self.var_half.get()} px')
            self._refresh()

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
            half = int(float(self.var_half.get()))
            px_nm = float(self.var_px.get())
            if self.var_source.get() == 'npy':
                if not self.tip_npy_path:
                    raise ValueError('尚未選擇探針 .npy（或改用廠商 cone）')
                tip = np.squeeze(np.load(self.tip_npy_path)).astype(np.float64)
                tip -= tip.max()
                return tip
            R = float(self.var_R.get())
            if self.var_asym.get():
                tx, ty = float(self.var_thx.get()), float(self.var_thy.get())
                return make_cone_tip_asym(half, px_nm, R, tx, ty)
            th = float(self.var_th.get())
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
            self._log('去卷積中…（grey_erosion + certainty）')

            def worker():
                try:
                    res = deconvolve(self.image, tip)
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror('去卷積失敗', str(e)))
                    return
                self.after(0, lambda: self._done(res))

            threading.Thread(target=worker, daemon=True).start()

        def _done(self, res):
            self.recon, self.certain = res['recon'], res['certain']
            self.frac = res['certain_frac']
            self.lbl_cert.config(text=f'{self.frac*100:.1f} %')
            self.lbl_din.config(text=f'{depth_of(self.image):.1f} nm')
            self.lbl_dre.config(text=f'{depth_of(self.recon):.1f} nm')
            # 下界性質檢查（誠實回報）
            ok = bool(np.all(self.recon <= self.image + 1e-6))
            self._log(f'完成：certain={self.frac*100:.1f}%  '
                      f'還原孔深={depth_of(self.recon):.1f}nm  '
                      f'下界 s_r≤i={ok}')
            self._log(f'  （紅色死角=探針碰不到、不可信；引擎只給確定下界不腦補）')
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
                        png, title=f'Certainty deconvolution — {stem}')
            self._log(f'已儲存至 {d}')
            messagebox.showinfo('完成', f'結果已存至：\n{d}')

        # ── 統一重繪（有什麼畫什麼）───────────────────────────
        def _refresh(self):
            self.fig.clf()
            axs = self.fig.subplots(2, 3)
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
            else:
                axs[0, 0].set_title('輸入影像（尚未載入）', fontsize=9)

            if self.recon is not None:
                axs[0, 1].imshow(self.recon, cmap='viridis', vmin=vmin, vmax=vmax)
                axs[0, 1].set_title(f'還原表面（erosion）\n'
                                    f'min={self.recon.min():.1f} nm 確定下界',
                                    fontsize=9)
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

            self.fig.tight_layout()
            self.canvas.draw()

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
    print(f"  影像孔深 = {depth_of(image):.1f} nm")
    print(f"  還原孔深 = {depth_of(recon):.1f} nm  (確定下界，不會過度去卷積)")
    print(f"  可信像素 certain = {res['certain_frac']*100:.1f}%"
          f"（其餘為探針碰不到的死角，僅下界）")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.out, f'{stem}_reconstructed.npy'), recon)
    np.save(os.path.join(args.out, f'{stem}_certain.npy'), res['certain'])
    save_panels(image, recon, res['certain'], tip, res['certain_frac'],
                os.path.join(args.out, f'{stem}_blind_deconv.png'),
                title=f'Certainty deconvolution — {stem}')
    print(f"\n完成。結果存於：{args.out}/")


if __name__ == '__main__':
    main()
