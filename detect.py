#!/usr/bin/env python3
"""
AFM 影像預測程式
使用訓練好的 Autoencoder 模型對新的 AFM 影像進行去卷積預測
"""

import os
import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.io import savemat
from scipy.ndimage import zoom

# 設定參數
OUTPUT_DIR = 'img'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'AFM_MAE_autoencoder.keras')
TARGET_SIZE = (128, 128)

# ---- 全域正規化常數（必須與訓練時完全相同）-----------------------
# 訓練腳本中設定：NORM_MIN=-155 nm, NORM_MAX=+105 nm
# detect.py 輸入影像須先以相同參數正規化後才能送入模型
# 模型輸出 [0,1] → 反正規化 → nm
NORM_MIN = -155.0   # nm
NORM_MAX = +105.0   # nm


def normalize_input(data_nm):
    """將 nm 影像正規化至 [0, 1]（與訓練時相同公式）。"""
    return (data_nm.astype('float32') - NORM_MIN) / (NORM_MAX - NORM_MIN)


def denormalize_output(data_01):
    """將模型輸出 [0, 1] 反正規化回 nm。"""
    return data_01.astype('float32') * (NORM_MAX - NORM_MIN) + NORM_MIN


# ══════════════════════════════════════════════════════════════
# Nanoscope 原始檔讀取（支援 .000 / .001 / .004 等副檔名）
# 與 afm_gui.py 的讀取邏輯相同
# ══════════════════════════════════════════════════════════════

def is_nanoscope_file(filepath):
    """判斷是否為 Nanoscope 原始掃描檔（副檔名為 3 位數字）。"""
    ext = os.path.splitext(filepath)[1]
    return bool(re.match(r'^\.\d{3}$', ext))


def parse_nanoscope_header(fp):
    """解析 Nanoscope 檔頭，回傳掃描參數。"""
    with open(fp, 'rb') as f:
        raw = f.read(65536)
    hdr = raw.decode('latin-1')

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
        'zsens': zs, 'zsens_s': zss, 'channels': chs,
    }


def read_nanoscope_channel(fp, meta, ch_idx=None):
    """
    讀取 Nanoscope 高度通道，回傳 nm 單位的 2D array。
    ch_idx=None → 自動選擇 Height Sensor 通道。
    """
    if ch_idx is None:
        ch_idx = 0
        for i, ch in enumerate(meta['channels']):
            if any(k.lower() in ch['name'].lower()
                   for k in ['ZSensor', 'Height Sensor', 'Height']):
                ch_idx = i
                break

    ch  = meta['channels'][ch_idx]
    n   = meta['n_px'] * meta['n_lines']
    dt  = np.int16 if ch['bpp'] == 2 else np.int32
    with open(fp, 'rb') as f:
        f.seek(ch['offset'])
        raw = np.frombuffer(f.read(n * ch['bpp']), dtype=dt)

    # 實際讀到的元素可能少於 n_px*n_lines（多通道檔案各通道行數可能不同）
    # → 從資料長度反推實際行數，zoom 會統一縮放至 128×128
    actual_lines = len(raw) // meta['n_px']
    if actual_lines == 0:
        raise ValueError(
            f"通道資料不足：只讀到 {len(raw)} 個元素，"
            f"無法組成 {meta['n_px']} px 寬的影像"
        )
    if actual_lines != meta['n_lines']:
        print(f"  [注意] 通道 '{ch['name']}' 實際行數 {actual_lines}"
              f"（header 記載 {meta['n_lines']}），以實際值為準")
    img  = raw[:actual_lines * meta['n_px']].reshape(
               actual_lines, meta['n_px']).astype(np.float64)
    sens = meta['zsens_s'] if 'ZsensSens' in ch['z_key'] else meta['zsens']
    nm_per_lsb = ch['z_lsb'] * sens
    print(f"  [Z校正] ch='{ch['name']}'  z_key={ch['z_key']}  "
          f"z_lsb={ch['z_lsb']:.6g}  sens={sens:.4f}  "
          f"→ {nm_per_lsb:.4f} nm/LSB")
    return img * nm_per_lsb


def flatten_rows_holes(img):
    """
    逐行線性基線校正，適用於孔洞型樣品（表面是高值）。
    每行去掉線性傾斜，然後以 percentile(95) 為基準歸零。
    """
    flat = img.copy()
    x    = np.arange(img.shape[1])
    for i in range(img.shape[0]):
        c = np.polyfit(x, flat[i], 1)
        flat[i] -= np.polyval(c, x)
    baseline = np.percentile(flat, 95)
    return flat - baseline


def load_nanoscope(filepath):
    """
    載入 Nanoscope 原始 AFM 掃描檔，套用行基線校正，
    resize 至 128×128，回傳 nm 單位的 2D array。
    """
    print(f"  讀取 Nanoscope 檔案：{filepath}")
    meta = parse_nanoscope_header(filepath)
    print(f"  掃描範圍：{meta['scan_nm']:.1f} nm  "
          f"解析度：{meta['n_px']}×{meta['n_lines']} px  "
          f"px_nm：{meta['px_nm']:.2f} nm/px")
    print(f"  可用通道：{[ch['name'] for ch in meta['channels']]}")

    img_nm = read_nanoscope_channel(filepath, meta)
    img_nm = flatten_rows_holes(img_nm)
    print(f"  基線校正後範圍：[{img_nm.min():.1f}, {img_nm.max():.1f}] nm")

    # Resize to 128×128（分別指定行、列縮放倍率，避免非正方形影像變形）
    h, w = img_nm.shape
    if (h, w) != (128, 128):
        img_nm = zoom(img_nm, (128 / h, 128 / w), order=3)

    # Z 範圍保護：若超出訓練正規化範圍 10 倍以上，自動縮放至合理範圍
    z_range = img_nm.max() - img_nm.min()
    train_range = NORM_MAX - NORM_MIN  # 260 nm
    if z_range > train_range * 10:
        scale = train_range / z_range
        img_nm = img_nm * scale
        print(f"  [Z保護] 偵測到異常 Z 範圍 {z_range:.1f} nm，"
              f"已縮放 ×{scale:.4f} → 新範圍 [{img_nm.min():.1f}, {img_nm.max():.1f}] nm")
        print(f"  ※ 建議檢查 Z 靈敏度設定（見上方 [Z校正] 輸出）")

    return img_nm.astype(np.float32)


def check_gpu():
    """檢查 GPU 是否可用。"""
    print("="*70)
    print("TensorFlow GPU 檢查")
    print("="*70)
    print(f"TensorFlow 版本: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU 可用: {gpus}")
    print(f"CUDA 建立: {tf.test.is_built_with_cuda()}")

    if gpus:
        print("✓ GPU 已啟用，將使用 GPU 進行預測")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠ GPU 不可用，將使用 CPU 進行預測")
    print("="*70 + "\n")


def preprocess_image(image_path):
    """
    預處理輸入影像。
    支援：
      - Nanoscope 原始掃描檔（.000 / .001 / .004 等，副檔名為 3 位數字）
      - 一般影像檔（.tif, .png, .jpg 等，像素值需為 nm 單位）

    Returns:
        processed_image : 正規化後的 (1, 128, 128, 1) array
        original_size   : 原始尺寸（像素或 (n_px, n_lines)）
        img_nm_2d       : 基線校正後 nm 單位的 128×128 2D array（用於顯示）
    """
    if is_nanoscope_file(image_path):
        # ---- Nanoscope 原始格式 --------------------------------
        img_nm = load_nanoscope(image_path)       # (128, 128) nm
        original_size = img_nm.shape

    else:
        # ---- 一般影像格式 --------------------------------------
        img = Image.open(image_path)
        original_size = img.size
        if img.mode != 'L':
            img = img.convert('L')
        img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img_nm = np.array(img_resized, dtype=np.float32)
        # 注意：一般圖像像素值為 0–255（灰階），非 nm
        # 若圖像已是 nm 單位可直接使用；否則需先轉換

    img_nm_2d = img_nm.copy()

    # ---- 套用全域正規化 → [0, 1] --------------------------------
    processed_image = normalize_input(img_nm)
    processed_image = processed_image.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 1)

    return processed_image, original_size, img_nm_2d


def predict(model, input_image):
    """
    使用模型進行預測

    Args:
        model: 載入的 Keras 模型
        input_image: 預處理後的輸入影像

    Returns:
        predicted_image: 預測結果
    """
    print("進行預測...")
    predicted_01 = model.predict(input_image, verbose=1)
    # 反正規化回 nm 單位
    predicted = denormalize_output(predicted_01)
    print(f"  預測輸出範圍（nm）: [{predicted.min():.2f}, {predicted.max():.2f}]")
    return predicted


def get_predict_dir(model_path):
    """
    根據模型路徑自動決定輸出目錄（YOLO 風格自動遞增）。

    runs/train7/weights/XXX.keras
        → runs/train7/predictions/predict1/   (首次)
        → runs/train7/predictions/predict2/   (再次)

    無法解析 trainN 時退回 img/predictions/predict{N}/
    """
    norm = os.path.normpath(model_path)
    parts = norm.split(os.sep)
    train_dir = None
    for i, p in enumerate(parts):
        if re.match(r'^train\d+$', p):
            train_dir = os.path.join(*parts[:i + 1])
            break

    base = os.path.join(train_dir, 'predictions') if train_dir \
           else os.path.join(OUTPUT_DIR, 'predictions')

    idx = 1
    while os.path.exists(os.path.join(base, f'predict{idx}')):
        idx += 1
    out = os.path.join(base, f'predict{idx}')
    os.makedirs(out, exist_ok=True)
    return out


def save_results(input_image, predicted_image, output_prefix, predict_dir):
    """
    儲存預測結果至指定目錄。

    Args:
        input_image   : 正規化後的輸入 (1,H,W,1)，值域 [0,1]
        predicted_image: 模型輸出（已反正規化至 nm）
        output_prefix : 檔名前綴（通常為輸入檔名的 stem）
        predict_dir   : 輸出目錄（由 get_predict_dir 決定）
    """
    # 提取資料 (移除 batch 和 channel 維度)
    input_2d     = denormalize_output(input_image)[0, :, :, 0]   # 反正規化回 nm
    predicted_2d = predicted_image[0, :, :, 0]

    def _stem(name):
        return os.path.join(predict_dir, f'{output_prefix}_{name}')

    # 儲存為 .npy
    np.save(_stem('input.npy'),     input_2d)
    np.save(_stem('predicted.npy'), predicted_2d)

    # 儲存為 .mat (MATLAB 格式)
    savemat(_stem('input.mat'),     {'image': input_2d})
    savemat(_stem('predicted.mat'), {'image': predicted_2d})

    # 儲存為影像 (.png) — 各自拉伸至 0-255
    def _to_u8(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

    Image.fromarray(_to_u8(input_2d)).save(_stem('input.png'))
    Image.fromarray(_to_u8(predicted_2d)).save(_stem('predicted.png'))

    print(f"\n結果已儲存至: {predict_dir}/")
    print(f"  - {output_prefix}_input.npy/.mat/.png")
    print(f"  - {output_prefix}_predicted.npy/.mat/.png")


def visualize_results(input_image, predicted_image, output_prefix,
                      predict_dir, no_display=False):
    """
    視覺化預測結果並儲存至 predict_dir。
    """
    input_2d     = denormalize_output(input_image)[0, :, :, 0]   # nm
    predicted_2d = predicted_image[0, :, :, 0]                   # nm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(input_2d, cmap='viridis')
    axes[0].set_title('Input (Dilated Image)  [nm]')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(predicted_2d, cmap='viridis')
    axes[1].set_title('Predicted (Deconvolved)  [nm]')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    save_path = os.path.join(predict_dir, f'{output_prefix}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  - {output_prefix}_comparison.png")

    if no_display:
        plt.close('all')
    else:
        plt.show()


def print_input_requirements():
    """印輸入影像要求說明。"""
    print("\n" + "="*70)
    print("輸入影像格式要求")
    print("="*70)
    print("""
模型期望的輸入影像：
┌─────────────────────────────────────────────────────────────────────┐
│  影像類型：AFM 掃描後的「膨脹影像」(Dilated Image)                     │
│                                                                     │
│  說明：這是經過 AFM 探針卷積後的影像，也就是實際掃描得到的影像。        │
│       模型會將這個影像「去卷積」，還原成真實的表面形貌。               │
│                                                                     │
│  支援格式：.tif, .tiff, .png, .jpg, .jpeg, .bmp                       │
│                                                                     │
│  建議前處理：                                                       │
│    1. 影像尺寸會自動調整為 128x128 像素                              │
│    2. 彩色影像會自動轉換為灰階                                       │
│    3. 數值範圍會自動進行背景減去 (0.0，新 tip 已無偏移)               │
│                                                                     │
│  範例：                                                             │
│    python detect.py C:\\path\\to\\your\\AFM_image.tif                  │
│    python detect.py ../analysis/AD-40-AS.tif                         │
└─────────────────────────────────────────────────────────────────────┘
""")
    print("="*70 + "\n")


def main():
    # 設定命令列參數解析
    parser = argparse.ArgumentParser(
        description='AFM 影像去卷積預測程式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python detect.py image.tif
  python detect.py C:/path/to/image.tif --output result
        """
    )
    parser.add_argument('image', nargs='?', help='輸入影像路徑 (.tif, .png, .jpg 等)')
    parser.add_argument('--output', '-o', default='result', help='輸出檔案前綴 (預設: result)')
    parser.add_argument('--model', '-m', default=MODEL_PATH, help='模型檔案路徑')
    parser.add_argument('--no-display', action='store_true', help='不顯示圖片')
    parser.add_argument('--info', '-i', action='store_true', help='顯示輸入影像要求說明')

    args = parser.parse_args()

    # 如果指定 --info 或沒有提供影像，顯示說明
    if args.info or args.image is None:
        print_input_requirements()
        if args.image is None and not args.info:
            parser.print_help()
        return

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.image):
        print(f"錯誤：找不到檔案 '{args.image}'")
        return

    # 檢查 GPU
    check_gpu()

    # 載入模型
    print("\n載入模型...")
    if not os.path.exists(args.model):
        print(f"錯誤：找不到模型檔案 '{args.model}'")
        print(f"請確認模型是否已訓練並儲存於: {MODEL_PATH}")
        return

    model = load_model(args.model)
    print(f"✓ 模型已載入: {args.model}\n")

    # ── YOLO 風格輸出目錄（根據模型路徑自動遞增）──────────────────
    predict_dir = get_predict_dir(args.model)
    # 以輸入檔名 stem 作為檔案前綴（若使用者有指定 -o 則優先）
    input_stem  = os.path.splitext(os.path.basename(args.image))[0]
    file_prefix = args.output if args.output != 'result' else input_stem
    print(f"輸出目錄：{predict_dir}/")
    print(f"檔案前綴：{file_prefix}\n")

    # 預處理輸入影像
    print(f"處理輸入影像: {args.image}")
    input_image, original_size, resized_original = preprocess_image(args.image)
    print(f"  原始尺寸: {original_size}")
    print(f"  調整後尺寸: {TARGET_SIZE}")
    print(f"  正規化後範圍: [{input_image.min():.3f}, {input_image.max():.3f}]  (應在 [0,1] 附近)")
    print()

    # 進行預測（內部自動反正規化回 nm）
    predicted_image = predict(model, input_image)
    print(f"✓ 預測完成！")
    print(f"  輸出範圍（nm）: [{predicted_image.min():.2f}, {predicted_image.max():.2f}]\n")

    # 儲存結果
    save_results(input_image, predicted_image, file_prefix, predict_dir)

    # 視覺化結果
    visualize_results(input_image, predicted_image, file_prefix,
                      predict_dir, no_display=args.no_display)

    print(f"\n✓ 全部完成！結果存於：{predict_dir}/")


if __name__ == '__main__':
    main()
