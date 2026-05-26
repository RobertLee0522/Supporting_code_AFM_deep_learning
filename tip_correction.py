#!/usr/bin/env python3
"""
tip_correction.py  —  AFM 探針方向校正程式
=====================================================
功能：
  1. 偵測探針頂端（最大值）位置，自動移至中心
  2. 套用徑向平均，強制旋轉對稱性
  3. Resize 至指定尺寸（預設 55×55）
  4. 儲存校正後的探針 .mat / .npy，並輸出視覺化圖

使用方式：
  python tip_correction.py tip.mat/tip_estimated0526.mat
  python tip_correction.py tip.mat/tip_estimated0526.mat --size 55
  python tip_correction.py tip.mat/tip_estimated0526.mat --size 55 --key tip --no-display

輸出：
  原始路徑目錄下，產生 <原檔名>_corrected.mat / <原檔名>_corrected.npy
  以及 <原檔名>_correction_report.png（四格診斷圖）
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage, io
from scipy.io import savemat


# ──────────────────────────────────────────────
# 核心函式
# ──────────────────────────────────────────────

def load_tip(mat_path, key='tip'):
    """讀取 .mat 檔，回傳 (raw_array, key_used)。"""
    data = io.loadmat(mat_path)
    if key in data:
        return data[key].astype(np.float64), key

    # 如果指定 key 不存在，自動尋找第一個二維陣列
    candidates = {k: v for k, v in data.items()
                  if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 2}
    if not candidates:
        raise KeyError(f"在 {mat_path} 中找不到二維陣列（嘗試 key='{key}'）\n"
                       f"可用 keys：{list(data.keys())}")
    found_key = next(iter(candidates))
    print(f"  [警告] key='{key}' 不存在，改用 '{found_key}'")
    return candidates[found_key].astype(np.float64), found_key


def diagnose(raw):
    """印出探針原始狀態診斷。"""
    max_val = raw.max()
    min_val = raw.min()
    max_idx = np.unravel_index(raw.argmax(), raw.shape)
    cy, cx  = raw.shape[0] // 2, raw.shape[1] // 2
    center_val = raw[cy, cx]
    off_center = np.sqrt((max_idx[0] - cy)**2 + (max_idx[1] - cx)**2)

    print(f"\n{'='*60}")
    print(f"  原始探針診斷")
    print(f"{'='*60}")
    print(f"  形狀          : {raw.shape[0]} × {raw.shape[1]} px")
    print(f"  最大值 (頂端) : {max_val:.4f} nm  位於 {max_idx}")
    print(f"  最小值        : {min_val:.4f} nm")
    print(f"  中心值        : {center_val:.4f} nm  位於 ({cy}, {cx})")
    print(f"  頂端偏離中心  : {off_center:.1f} px")
    if off_center > 3:
        print(f"  [!] 頂端明顯偏離中心 → 需要方向校正")
    else:
        print(f"  [OK] 頂端已在中心附近")

    # 徑向對稱性評估（從中心算）
    ny, nx = raw.shape
    y_idx, x_idx = np.indices((ny, nx))
    r_map = np.round(np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)).astype(int)
    r_std_list = []
    for ri in range(1, min(cy, cx)):
        mask = r_map == ri
        if mask.sum() >= 3:
            r_std_list.append(raw[mask].std())
    mean_r_std = np.mean(r_std_list) if r_std_list else 0
    print(f"  徑向標準差均值: {mean_r_std:.3f} nm  (越小越對稱)")
    print(f"{'='*60}\n")
    return off_center, mean_r_std


def center_apex(raw):
    """步驟 1：將最大值（頂端）移至中心。"""
    max_idx = np.unravel_index(raw.argmax(), raw.shape)
    cy, cx  = raw.shape[0] // 2, raw.shape[1] // 2
    shifted = ndimage.shift(raw, [cy - max_idx[0], cx - max_idx[1]], mode='nearest')
    shifted -= shifted.max()   # 確保 max = 0
    return shifted, max_idx


def radial_average(tip_2d):
    """步驟 2：套用徑向平均，強制旋轉對稱。"""
    ny, nx  = tip_2d.shape
    cy, cx  = ny // 2, nx // 2
    y_idx, x_idx = np.indices((ny, nx))
    r_map   = np.round(np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)).astype(int)
    out     = np.zeros_like(tip_2d)
    for ri in range(r_map.max() + 1):
        mask = r_map == ri
        if mask.any():
            out[mask] = tip_2d[mask].mean()
    out -= out.max()            # 確保 max = 0
    return out


def resize_tip(tip_2d, target_size):
    """步驟 3：Resize 至 target_size × target_size。"""
    zoom_factor = target_size / tip_2d.shape[0]
    resized = ndimage.zoom(tip_2d, zoom_factor, order=3)
    resized -= resized.max()    # 插值後再次確保 max = 0
    return resized


def radial_profile(tip_2d):
    """計算從中心出發的徑向剖面（用於繪圖）。"""
    ny, nx  = tip_2d.shape
    cy, cx  = ny // 2, nx // 2
    y_idx, x_idx = np.indices((ny, nx))
    r_map   = np.round(np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)).astype(int)
    r_max   = min(cy, cx)
    r_vals, mean_vals = [], []
    for ri in range(r_max + 1):
        mask = r_map == ri
        if mask.any():
            r_vals.append(ri)
            mean_vals.append(tip_2d[mask].mean())
    return np.array(r_vals), np.array(mean_vals)


def correct_tip(mat_path, key='tip', target_size=55):
    """
    完整校正流程：載入 → 診斷 → 中心化 → 徑向平均 → Resize。
    回傳 (raw, after_center, after_radial, after_resize)。
    """
    print(f"\n[1/4] 載入探針：{mat_path}")
    raw, used_key = load_tip(mat_path, key)

    print(f"[2/4] 診斷原始探針狀態...")
    off_center, r_std = diagnose(raw)

    print(f"[3/4] 執行方向校正...")

    # Step A-1：移頂端至中心
    centered, orig_max_idx = center_apex(raw)
    cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
    print(f"  頂端位移：{orig_max_idx} → ({cy}, {cx})")

    # Step A-2：徑向平均
    radial_avg = radial_average(centered)

    # Step B：Resize
    resized = resize_tip(radial_avg, target_size)
    print(f"  Resize：{radial_avg.shape[0]}×{radial_avg.shape[1]}"
          f" → {resized.shape[0]}×{resized.shape[1]}")
    print(f"  校正後數值範圍：[{resized.min():.3f}, {resized.max():.3f}] nm")
    print(f"[4/4] 校正完成。\n")

    return raw, centered, radial_avg, resized, used_key


# ──────────────────────────────────────────────
# 視覺化
# ──────────────────────────────────────────────

def visualize(raw, centered, radial_avg, resized, out_path):
    """
    四格診斷圖：
      [左上] 原始探針           [右上] 中心化後
      [左下] 徑向平均後         [右下] 最終 (resize)
    加上右側徑向剖面比較圖。
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.1], wspace=0.35, hspace=0.4)

    panels = [
        (raw,        "1. 原始 (Raw)",              gs[0, 0]),
        (centered,   "2. 中心化 (Apex → Center)",  gs[0, 1]),
        (radial_avg, "3. 徑向平均 (Radial Avg)",   gs[1, 0]),
        (resized,    f"4. 最終 ({resized.shape[0]}×{resized.shape[1]} Resized)",  gs[1, 1]),
    ]

    vmin = raw.min()
    vmax = 0.0

    for data, title, pos in panels:
        ax = fig.add_subplot(pos)
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='nm')

    # 徑向剖面比較
    ax_prof = fig.add_subplot(gs[:, 2])
    for data, label, ls in [
        (raw,        "原始（從中心）",     '--'),
        (centered,   "中心化後",           '-'),
        (radial_avg, "徑向平均後",         '-'),
    ]:
        r, v = radial_profile(data)
        ax_prof.plot(r, v, linestyle=ls, label=label)

    # 校正後 resized 的剖面（以 55px 尺度）
    r_r, v_r = radial_profile(resized)
    # 換算回原始 px 尺度以便比較
    scale = raw.shape[0] / resized.shape[0]
    ax_prof.plot(r_r * scale, v_r, linestyle='-', linewidth=2, label=f"Resized（×{scale:.2f}）")

    ax_prof.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax_prof.set_xlabel("距中心距離 (px，原始尺度)")
    ax_prof.set_ylabel("高度 (nm)")
    ax_prof.set_title("徑向剖面比較")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle("AFM 探針校正診斷報告", fontsize=13, fontweight='bold')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  診斷圖已儲存：{out_path}")
    plt.show()


# ──────────────────────────────────────────────
# 儲存
# ──────────────────────────────────────────────

def save_corrected(resized, mat_path, used_key):
    """
    在原始 .mat 同目錄下，另存 <原檔名>_corrected.mat 與 .npy。
    .mat 內保留 key='tip'（訓練程式使用）及 'px_nm' 欄位說明。
    """
    base      = os.path.splitext(mat_path)[0]
    out_mat   = base + '_corrected.mat'
    out_npy   = base + '_corrected.npy'

    savemat(out_mat, {'tip': resized, 'original_key': used_key,
                      'shape': list(resized.shape),
                      'note': 'corrected: apex centered + radial avg + resized'})
    np.save(out_npy, resized)

    print(f"  校正結果已儲存：")
    print(f"    {out_mat}")
    print(f"    {out_npy}")
    return out_mat, out_npy


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AFM 探針方向校正程式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
校正流程說明：
  Step 1  偵測最大值（探針頂端）位置
  Step 2  將頂端移至影像中心（scipy.ndimage.shift）
  Step 3  套用徑向平均，強制旋轉對稱（消除四重對稱假象）
  Step 4  Resize 至指定尺寸（預設 55×55）
  Step 5  確保最終 max = 0（AFM 探針慣例：頂端 = 0 nm）

範例：
  python tip_correction.py tip.mat/tip_estimated0526.mat
  python tip_correction.py tip.mat/tip_estimated0526.mat --size 55
  python tip_correction.py tip.mat/tip_estimated0526.mat --size 55 --key tip --no-display
        """
    )
    parser.add_argument('tip_file',
                        help='輸入探針 .mat 檔案路徑')
    parser.add_argument('--key', '-k', default='tip',
                        help='mat 檔中的陣列 key（預設: tip）')
    parser.add_argument('--size', '-s', type=int, default=55,
                        help='輸出探針尺寸 px（預設: 55）')
    parser.add_argument('--no-display', action='store_true',
                        help='不顯示圖片（僅儲存）')
    parser.add_argument('--diagnose-only', '-d', action='store_true',
                        help='只診斷，不儲存校正結果')

    args = parser.parse_args()

    # 確認輸入檔案存在
    if not os.path.exists(args.tip_file):
        print(f"錯誤：找不到檔案 '{args.tip_file}'")
        sys.exit(1)

    # 執行校正
    raw, centered, radial_avg, resized, used_key = correct_tip(
        args.tip_file, key=args.key, target_size=args.size
    )

    # 校正後診斷
    print("校正後探針狀態：")
    cy, cx = resized.shape[0] // 2, resized.shape[1] // 2
    print(f"  形狀         : {resized.shape[0]} × {resized.shape[1]} px")
    print(f"  中心值 (頂端): {resized[cy, cx]:.6f} nm  ← 應為 0")
    print(f"  最大值位置   : {np.unravel_index(resized.argmax(), resized.shape)}")
    print(f"  最小值 (底部): {resized.min():.3f} nm")

    # 徑向剖面列印
    r_vals, mean_vals = radial_profile(resized)
    print(f"\n  徑向剖面（校正後 {resized.shape[0]}×{resized.shape[1]}）：")
    checkpoints = [0, 2, 5, 10, 15, 20, 25, cy]
    for rr in checkpoints:
        if rr < len(r_vals):
            print(f"    r={rr:2d} px → {mean_vals[rr]:8.3f} nm")

    # 視覺化
    base     = os.path.splitext(args.tip_file)[0]
    fig_path = base + '_correction_report.png'
    if not args.no_display or not args.diagnose_only:
        plt.rcParams['font.family'] = 'sans-serif'
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'DejaVu Sans']
        except Exception:
            pass

    visualize(raw, centered, radial_avg, resized, fig_path)

    if args.diagnose_only:
        print("\n[diagnose-only 模式] 未儲存校正結果。")
        return

    # 儲存
    out_mat, out_npy = save_corrected(resized, args.tip_file, used_key)

    print(f"""
{'='*60}
  完成！校正後的探針可在訓練程式中這樣使用：

  tip_shape = load_and_prepare_tip('{out_mat}')
  # 或直接用：
  tip_shape = np.load('{out_npy}')
  tip_shape -= tip_shape.max()   # 確保 max=0
{'='*60}
""")


if __name__ == '__main__':
    main()
