#!/usr/bin/env python3
"""
AFM 影像預測程式
使用訓練好的 Autoencoder 模型對新的 AFM 影像進行去卷積預測

# 自動以輸入檔名（std.000）當前綴
python detect.py datafortip/forpredict/std.000 -m runs/train7/weights/AFM_MAE_autoencoder.keras

# 自訂前綴
python detect.py std.004 -m runs/train7/weights/AFM_MAE_autoencoder.keras -o exp1_std004
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
TARGET_SIZE = (128, 128)


def find_latest_model():
    """
    在 runs/train{N}/weights/ 中找到編號最大的已訓練模型。
    找不到時回傳 None（交由 argparse 預設值處理）。
    """
    runs_dir = 'runs'
    if not os.path.exists(runs_dir):
        return None
    best_idx, best_path = -1, None
    for d in os.listdir(runs_dir):
        m = re.match(r'^train(\d+)$', d)
        if not m:
            continue
        idx = int(m.group(1))
        candidate = os.path.join(runs_dir, d, 'weights',
                                 'AFM_MAE_autoencoder.keras')
        if os.path.exists(candidate) and idx > best_idx:
            best_idx, best_path = idx, candidate
    return best_path


MODEL_PATH = find_latest_model() or os.path.join(OUTPUT_DIR, 'AFM_MAE_autoencoder.keras')

# ---- 全域正規化常數（必須與訓練時完全相同）-----------------------
# 訓練腳本中設定：NORM_MIN=-155 nm, NORM_MAX=+105 nm
# detect.py 輸入影像須先以相同參數正規化後才能送入模型
# 模型輸出 [0,1] → 反正規化 → nm
NORM_MIN = -175.0   # nm（與訓練腳本一致；真實掃描最低可達 -160nm）
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
    """判斷是否為 Nanoscope 原始掃描檔（副檔名以 3 位數字開頭，
    如 .000、.004；亦容許後接描述文字，如 .000-after_flatten，
    方便使用者手動標記處理過的檔案而不改變底層二進位格式）。"""
    ext = os.path.splitext(filepath)[1]
    return bool(re.match(r'^\.\d{3}', ext))


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
        dln = re.search(r'\\Data length:\s*(\d+)', blk)
        nm  = re.search(r'@2:Image Data:\s*S\s*\[([^\]]+)\].*?"([^"]+)"', blk)
        zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
        if off and nm:
            chs.append({
                'offset':      int(off.group(1)),
                'bpp':         int(bpp.group(1)) if bpp else 2,
                # 部分掃描（中斷/未跑滿）時，channel 的實際有效資料量
                # 會小於 n_px×n_lines×bpp；務必以此欄位為界，避免讀到
                # 下一個 channel 的資料區（見 detect.py changelog 2026-06-16）
                'data_length': int(dln.group(1)) if dln else None,
                'key':         nm.group(1).strip(),
                'name':        nm.group(2).strip(),
                'z_key':       zsc.group(1) if zsc else 'ZsensSens',
                'z_lsb':       float(zsc.group(2)) if zsc else 0.000375,
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
    read_bytes = n * ch['bpp']
    # 部分掃描（中斷/未跑滿）時，channel 宣告的 Data length 會小於
    # n_px×n_lines×bpp；務必以此為界，否則會讀到下一個 channel 的資料區
    # （曾發生 Height Sensor 因此混入 DMTModulus/Adhesion 等通道數值，
    #   造成 std 異常大、看似雜訊但實為跨通道污染）
    if ch.get('data_length'):
        read_bytes = min(read_bytes, ch['data_length'])
    with open(fp, 'rb') as f:
        f.seek(ch['offset'])
        raw = np.frombuffer(f.read(read_bytes), dtype=dt)

    # 實際讀到的元素可能少於 n_px*n_lines（多通道檔案各通道行數可能不同，
    # 或整次掃描中途中斷）→ 從資料長度反推實際行數，zoom 會統一縮放至 128×128
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


def despike_image(img, win=3, abs_thresh_nm=300.0, n_sigma=8.0):
    """
    去除 AFM 掃描尖刺（spike）與孤立壞線。

    原理：真實表面形貌在空間上連續——孤立像素若與「局部中位數」差距
    過大，多為探針碰撞、回授失鎖或通道資料錯位造成的非物理尖刺。以 3×3 中值
    濾波估計局部基準，殘差超過門檻者以局部中值取代。

    為何不會誤刪真實深孔：深孔/壞線的鄰域同樣偏移，中值也跟著偏移 → 殘差小，
    不會被標記；只有「孤立且與鄰域差距巨大」的尖刺才被移除。

    門檻 = max(abs_thresh_nm, n_sigma × 穩健標準差)。abs_thresh_nm 來自物理上限：
    本樣品孔洞深度 < ~300nm，單像素偏離局部中位數 >300nm 必為尖刺（如 -5400nm）。
    """
    from scipy.ndimage import median_filter
    med   = median_filter(img, size=win)
    resid = img - med
    mad   = np.median(np.abs(resid - np.median(resid)))
    robust_sigma = 1.4826 * mad
    thresh = max(abs_thresh_nm, n_sigma * robust_sigma)
    mask   = np.abs(resid) > thresh
    n_spike = int(mask.sum())
    if n_spike > 0:
        img = img.copy()
        img[mask] = med[mask]
        print(f"  [去尖刺] 移除 {n_spike} 個尖刺像素"
              f"（門檻 {thresh:.0f} nm，以局部中值取代）"
              f" → 範圍 [{img.min():.1f}, {img.max():.1f}] nm")
    return img


def clip_physical_range(img, max_depth_nm=300.0, max_height_nm=60.0):
    """
    將孔洞型 AFM 影像裁切到物理合理的 Z 窗（相對表面中位數）。

    用途：處理「連續壞區」——3×3 中值濾波去尖刺無法移除的大面積非物理區塊，
    例如部分掃描（行數不足）、回授失鎖或通道資料錯位造成的數 µm 假台階。
    若不裁切，這類壞區會污染 flatten 的 percentile(95) 基準，並觸發 Z 全圖縮放、
    把真實孔洞壓扁。

    物理假設：孔洞樣品的表面是高基準（≈中位數），真實特徵向下凹；故高側收緊
    （表面之上幾乎無真實結構，只有 ±20nm 掃描條紋雜訊），低側放寬以保留深孔。
    壞區被夾到表面附近 → 視為平坦表面（無意義但無害），真實孔洞完整保留。
    """
    ref = float(np.median(img))
    lo, hi = ref - max_depth_nm, ref + max_height_nm
    n_clip = int(np.sum((img < lo) | (img > hi)))
    if n_clip > 0:
        img = np.clip(img, lo, hi)
        print(f"  [物理裁切] 表面中位數基準 {ref:.1f} nm；裁切 {n_clip} 個越界像素"
              f"至 [{lo:.0f}, {hi:.0f}] nm（移除連續壞區/大台階）"
              f" → 範圍 [{img.min():.1f}, {img.max():.1f}] nm")
    return img


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
    img_nm = despike_image(img_nm)        # ① 去孤立尖刺（中值濾波殘差門檻）
    img_nm = clip_physical_range(img_nm)  # ② 裁切連續壞區/大台階（中值無法移除者）
    img_nm = flatten_rows_holes(img_nm)
    print(f"  基線校正後範圍：[{img_nm.min():.1f}, {img_nm.max():.1f}] nm")

    # Resize to 128×128（分別指定行、列縮放倍率，避免非正方形影像變形）
    h, w = img_nm.shape
    if (h, w) != (128, 128):
        img_nm = zoom(img_nm, (128 / h, 128 / w), order=3)

    # ── 橫向尺度診斷 ──────────────────────────────────────────────
    # 訓練假設：每 px = SURFACE_SCALE_NM = 39.1 nm（5000nm 掃描 / 128px）
    # 若真實掃描範圍不同，resize 後每 px 物理距離會偏移，導致孔洞被模型誤判尺寸
    SURFACE_SCALE_NM = 39.1   # 訓練時的基準（nm/px）
    actual_px_nm = meta['scan_nm'] / 128.0
    scale_ratio  = actual_px_nm / SURFACE_SCALE_NM
    print(f"\n  ── 橫向尺度檢查 ──")
    print(f"  掃描範圍     : {meta['scan_nm']:.1f} nm")
    print(f"  實際 px_nm   : {actual_px_nm:.2f} nm/px  (resize 至 128×128 後)")
    print(f"  訓練 px_nm   : {SURFACE_SCALE_NM:.2f} nm/px  (5000nm / 128px)")
    if abs(scale_ratio - 1.0) > 0.15:
        print(f"  ⚠ 尺度偏差 {(scale_ratio-1)*100:+.1f}%：孔洞在像素空間偏大/偏小 {scale_ratio:.2f}×")
        print(f"    → 建議將掃描範圍調整為 5000nm，或重新訓練匹配當前掃描尺度")
    else:
        print(f"  ✓ 尺度相符（偏差 {(scale_ratio-1)*100:+.1f}%）")

    # ── Z 範圍保護（兩段：嚴重異常縮放 + 超出分布軟性警告）────────
    z_range    = img_nm.max() - img_nm.min()
    train_range = NORM_MAX - NORM_MIN      # 280 nm
    print(f"\n  ── Z 範圍檢查 ──")
    print(f"  輸入 Z 範圍  : [{img_nm.min():.1f}, {img_nm.max():.1f}] nm  (總跨度 {z_range:.1f} nm)")
    print(f"  訓練分布範圍 : [{NORM_MIN:.1f}, {NORM_MAX:.1f}] nm")

    if z_range > train_range * 10:
        scale = train_range / z_range
        img_nm = img_nm * scale
        print(f"  ⚠ Z 保護觸發：異常範圍 {z_range:.1f} nm，縮放 ×{scale:.4f} → "
              f"[{img_nm.min():.1f}, {img_nm.max():.1f}] nm")
        print(f"    建議檢查 Z 靈敏度（見上方 [Z校正] 輸出）")
    elif img_nm.min() < NORM_MIN or img_nm.max() > NORM_MAX:
        print(f"  ⚠ 部分 Z 值超出訓練分布！超出範圍的像素數："
              f"  低端={np.sum(img_nm < NORM_MIN)}  高端={np.sum(img_nm > NORM_MAX)}")
        print(f"    → 預測結果可能不穩定，建議確認 Z 靈敏度設定")
    else:
        print(f"  ✓ Z 範圍在訓練分布內")

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
    左右兩張使用相同色階（vmin/vmax），才能正確比較孔洞深淺變化。
    """
    input_2d     = denormalize_output(input_image)[0, :, :, 0]   # nm
    predicted_2d = predicted_image[0, :, :, 0]                   # nm

    # ── 統計資訊（判斷去卷積是否合理）─────────────────────────────
    in_min,  in_max  = input_2d.min(),  input_2d.max()
    pr_min,  pr_max  = predicted_2d.min(), predicted_2d.max()
    print(f"\n  ── 去卷積結果比較 ──")
    print(f"  輸入孔洞深度  (min nm) : {in_min:.1f} nm")
    print(f"  預測孔洞深度  (min nm) : {pr_min:.1f} nm")
    # depth_ratio = pr_min / in_min（兩值均為負數時）
    # 兩者皆負：ratio > 1.0 → |pr| > |in| → 預測更深（去卷積正確方向）
    #           ratio < 1.0 → |pr| < |in| → 預測更淺（方向錯誤 / 輸入資料髒）
    #           ratio > 3.0 → 過度去卷積
    depth_ratio = pr_min / in_min if in_min != 0 else float('inf')
    if 1.0 <= depth_ratio < 3.0:
        print(f"  ✓ 深度比 {depth_ratio:.2f}×（|預測| > |輸入|，孔洞加深，符合去卷積物理）")
    elif depth_ratio >= 3.0:
        print(f"  ⚠ 深度比 {depth_ratio:.2f}× 偏大，可能過度去卷積")
        print(f"    常見原因：① 掃描範圍不是 5000nm ② Z靈敏度錯誤 ③ 訓練探針與實際不符")
    elif depth_ratio > 0.0:
        print(f"  ⚠ 深度比 {depth_ratio:.2f}× < 1（|預測| < |輸入|，孔洞被做淺）")
        print(f"    常見原因：① 輸入含大片壞區使模型判斷背景錯誤 ② 輸入 Z 超出訓練分布")
    else:
        print(f"  ⚠ 深度比 {depth_ratio:.2f}×，方向異常（預測與輸入同號或 in_min 近零）")

    # ── 共用色階（防止視覺誤判）─────────────────────────────────
    # 使用兩張圖的聯集範圍，確保顏色對比有意義
    vmin = min(in_min, pr_min)
    vmax = max(in_max, pr_max)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(input_2d, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Input (AFM scan)  [nm]\nmin={in_min:.1f}  max={in_max:.1f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(predicted_2d, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Predicted (Deconvolved)  [nm]\nmin={pr_min:.1f}  max={pr_max:.1f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 差值圖（顯示模型做了多少校正）
    diff = predicted_2d - input_2d
    im3 = axes[2].imshow(diff, cmap='RdBu_r',
                         vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes[2].set_title(f'Correction (Predicted - Input)  [nm]\n'
                      f'max correction = {diff.min():.1f} nm')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 用英文標題避免無 CJK 字型環境的 glyph-missing 警告與方框亂碼
    plt.suptitle('AFM Deconvolution (shared color scale for valid comparison)',
                 fontsize=11)
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
