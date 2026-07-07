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

用法：
  python blind_deconvolution.py --demo                       # 合成資料驗證 + 教學圖
  python blind_deconvolution.py scan.npy --tip tip.npy       # 用已知探針去卷積
  python blind_deconvolution.py scan.npy --cone-R 2 --cone-theta 25 --tip-half 5
                                                             # 用廠商 cone 去卷積
（輸入 .npy 為 nm 單位 2D 陣列；Nanoscope 原始檔可先用 detect.py 轉出 _input.npy）
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # 無頭環境輸出 PNG
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, grey_erosion


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


def make_paraboloid_tip(half_px, px_nm, R_nm=46.9):
    """拋物面探針 z(r) = −r²/(2R)（apex=0）。較鈍 → 明顯失真孔洞，適合 demo。"""
    size = 2 * half_px + 1
    yi, xi = np.indices((size, size))
    r = np.sqrt((yi - half_px) ** 2 + (xi - half_px) ** 2) * px_nm
    tip = -(r ** 2) / (2.0 * R_nm)
    tip -= tip.max()
    return tip


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
# CLI
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='通用 AFM 探針去卷積（Villarrubia 形態學 certainty，零訓練）')
    ap.add_argument('image', nargs='?', help='輸入影像 .npy（nm 單位 2D 陣列）')
    ap.add_argument('--tip', help='已知探針 .npy（apex 置中=0，向外為負）')
    ap.add_argument('--cone-R', type=float, help='改用廠商 cone：尖端球半徑 R (nm)')
    ap.add_argument('--cone-theta', type=float, default=25.0, help='cone 半錐角 θ (°)')
    ap.add_argument('--tip-half', type=int, default=5, help='cone 視窗半徑 px')
    ap.add_argument('--px-nm', type=float, default=39.1, help='影像每 px nm（預設 39.1）')
    ap.add_argument('--out', default='blind_out', help='輸出資料夾')
    ap.add_argument('--demo', action='store_true', help='跑合成資料驗證 + 教學圖')
    args = ap.parse_args()

    if args.demo or args.image is None:
        if args.image is None and not args.demo:
            ap.print_help()
            print("\n（提示：先試 `python blind_deconvolution.py --demo`）")
            return
        run_demo('blind_demo')
        return

    os.makedirs(args.out, exist_ok=True)
    image = np.squeeze(np.load(args.image)).astype(np.float64)
    print(f"輸入影像：{args.image}  shape={image.shape}  "
          f"範圍[{image.min():.1f}, {image.max():.1f}] nm")

    if args.tip:
        tip = np.load(args.tip).astype(np.float64); tip -= tip.max()
        print(f"探針：載入 {args.tip}  shape={tip.shape}")
    elif args.cone_R is not None:
        tip = make_cone_tip(args.tip_half, args.px_nm, args.cone_R, args.cone_theta)
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
