# ============================================================
# AFM Deep Learning - Supporting Code (Fixed Version)
# 修正清單：
#   [Fix 1]  spherical image_creator 重新命名為 spherical_image_creator，
#            避免被 cubic 版本覆蓋
#   [Fix 2]  circle_randomizer / cubic_randomizer 加入最大嘗試次數，
#            防止無限迴圈
#   [Fix 3]  cubic_randomizer 非重疊判斷改用 abs()，邏輯更清晰
#   [Fix 4]  image_creator (spherical & cubic) 修正 arange 從 0 開始、
#            迴圈涵蓋全部像素 (0..size-1)，不再缺漏邊緣像素
#   [Fix 5]  line_artifact_adder 方向修正：
#            (128,1) → (1,128)，改為模擬水平 row-based AFM 掃描偽影
#   [Fix 6]  模型存檔副檔名統一為 .keras；
#            所有 load_model 呼叫也改為 .keras
#   [Fix 7]  移除未使用的 input_img = Input(...) 及 nb_classes = 2
#   [Fix 8]  TrainingProgressCallback 移除無意義的 "100 - MAE" 準確率，
#            改顯示相對改善率 (val_loss / train_loss)
#   [Fix 9]  calculate_metrics 中的 SSIM 改用
#            skimage.metrics.structural_similarity 計算正確值
#   [Fix 10] 路徑變數命名修正：
#            training_images_path → dilated_images_path (X 輸入)
#            labels_path         → true_images_path    (y 標籤)
#   [Fix 11] decoded_imgs 保持為 numpy array，不轉換成 list
#   [Fix 12] 各 plt.subplots 補上 plt.tight_layout() 與 plt.show()
# ============================================================

# Import required modules

# Data loading, plotting, and dataset generation
import numpy as np
import pandas as pd
import os
import math
import random
from math import sqrt
import matplotlib
matplotlib.use('Agg')          # 不開視窗，直接存檔
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy import io
from skimage.draw import random_shapes
from skimage.metrics import structural_similarity as ssim_func  # [Fix 9]
from PIL import Image

# Dataset manipulation
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Deep learning model
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import (Activation, Input, Convolution2D, MaxPooling2D,
                           UpSampling2D, Conv2DTranspose, Concatenate)
from keras.callbacks import Callback, History

# Progress bar and metrics display
from tqdm import tqdm
import time

# Data saving
from scipy.io import savemat

# ============================================================
# 輸出目錄（中間資料）
# ============================================================
OUTPUT_DIR = 'img'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# YOLO 風格自動遞增訓練目錄
#   runs/train1/  →  runs/train2/  →  ...
#   weights/   存放 .keras / .weights.h5
# ============================================================
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)
_run_idx = 1
while os.path.exists(os.path.join(RUNS_DIR, f'train{_run_idx}')):
    _run_idx += 1
RUN_DIR     = os.path.join(RUNS_DIR, f'train{_run_idx}')
WEIGHTS_DIR = os.path.join(RUN_DIR, 'weights')
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
print(f"\n{'='*60}")
print(f"  本次訓練目錄：{RUN_DIR}")
print(f"{'='*60}\n")

# 圖片編號計數器（確保檔名排序正確）
_fig_counter = [0]

def save_plot(filename, dpi=150):
    """儲存目前 figure 至本次訓練目錄並關閉，同時印出路徑。"""
    _fig_counter[0] += 1
    numbered = f'{_fig_counter[0]:02d}_{filename}'
    path = os.path.join(RUN_DIR, numbered)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    print(f"  [圖片] {path}")

# ============================================================
# 探針前處理常數
# ============================================================
SURFACE_SCALE_NM = 39.1  # nm/px：對應真實掃描 5000nm ÷ 128px
#   此值必須與 detect.py 推論掃描的 px_nm 一致：
#   5000nm 掃描 / 256px → resize 128px = 5000/128 = 39.1 nm/px
TARGET_TIP_SIZE  = 55    # 將估算探針 resize 至此尺寸 (px)（供 tip_correction 使用）

# ---- 訓練用合成探針設定 ----------------------------------------
# ROC 從 tip_estimated0526.mat 量得：R ≈ 73 nm
#
# TIP_TRAIN_SIZE = 5 px（at 39.1 nm/px）：
#   有效探針半徑 = 46.9 nm（afm_gui 量得）/ 39.1 nm/px ≈ 1.2 px
#   有效填滿半徑 = sqrt(2 × 46.9 × 125) / 39.1 ≈ 2.77 px（對 125 nm 深孔）
#   真實孔洞半徑 ≈ 2.93 px → 兩者相近，孔洞在訓練中不應被 100% 填平
#   舊版使用 73nm/9px，tip 半寬 3px × 39.1 = 117nm > 孔洞半徑 114nm →完全填平 → 過度去卷積
#   修正後：tip 半寬 2px × 39.1 = 78nm < 孔洞半徑 114nm → 孔洞在訓練中可見
# ---------------------------------------------------------------
TIP_RADIUS_NM  = 46.9   # afm_gui 量得真實探針 ROC (nm)；比舊值 73.0 更尖
TIP_TRAIN_SIZE = 5      # 合成探針尺寸 (px)；縮小至 5px 避免訓練時孔洞被完全填平

# ---- 梯形孔樣品標稱參數（用於訓練資料生成）--------------------
# 來源：用戶實際掃描的校正樣品
TRAP_OPEN_NM  = 228.8   # 開口寬度 (nm)
TRAP_BOT_NM   = 183.5   # 底部寬度 (nm)
TRAP_DEPTH_NM = 125.8   # 深度 (nm)
TRAP_VARIATION = 0.20   # 隨機變異範圍 ±20%

# ---- 全域正規化常數（解決模型輸出全零的關鍵）---------------------
# 訓練資料物理範圍：
#   負值：孔洞最深 -145nm + 掃描條紋 ±20nm + 雜訊 ≈ -167nm
#         → 取 -175 nm 留足夠安全邊界（避免真實掃描超出分布範圍）
#   正值：表面最高 = 取 105 nm（含安全邊界）
# 公式：x_norm = (x - NORM_MIN) / (NORM_MAX - NORM_MIN)  → [0, 1]
# 效果：背景 0 nm 正規化後 = 0.625（不再是特殊零值）
#       孔洞 -145 nm → 0.107；模型無法靠「全零輸出」最小化 loss
# ---------------------------------------------------------------
NORM_MIN = -175.0   # nm（含安全邊界；真實掃描最低可達 -160nm）
NORM_MAX = +105.0   # nm（含安全邊界）


def normalize_data(data):
    """將 nm 資料正規化至 [0, 1]。"""
    return (data.astype('float32') - NORM_MIN) / (NORM_MAX - NORM_MIN)


def denormalize_data(data):
    """將 [0, 1] 預測結果還原至 nm 單位。"""
    return data.astype('float32') * (NORM_MAX - NORM_MIN) + NORM_MIN


def make_training_tip(size=TIP_TRAIN_SIZE, radius_nm=TIP_RADIUS_NM,
                      px_nm=SURFACE_SCALE_NM):
    """
    建立拋物線形訓練探針：z(r) = -r² / (2R)

    - 中心 = 頂端 = 0 nm（最大值，符合 grey_dilation 慣例）
    - 向外遞減（越負越深）
    - 在邊緣 (r = size//2 * px_nm nm) 的深度 = -(edge_nm)²/(2R)
      若此值 < 任何粒子高度 → 粒子不會擴散出 tip 邊緣 → 不飽和

    Args:
        size      : 探針 px 數（奇數）
        radius_nm : 曲率半徑 (nm)，越小越尖
        px_nm     : 每像素 nm 數（應與 SURFACE_SCALE_NM 一致）
    Returns:
        tip_2d : ndarray shape (size, size), max=0 at center
    """
    if size % 2 == 0:
        size += 1            # 確保中心像素存在
    cy, cx = size // 2, size // 2
    y_idx, x_idx = np.indices((size, size))
    r_nm   = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2) * px_nm
    tip_2d = -(r_nm ** 2) / (2.0 * radius_nm)
    tip_2d -= tip_2d.max()  # 確保 max = 0（浮點安全）

    edge_depth = tip_2d[0, 0]
    print(f"  [訓練探針] size={size}×{size} px, R={radius_nm} nm, "
          f"px_nm={px_nm} nm/px")
    print(f"  邊緣深度 = {edge_depth:.1f} nm  "
          f"（粒子高度需 < {abs(edge_depth):.0f} nm 才不飽和）")
    return tip_2d


# ============================================================
# 梯形孔 (Trapezoidal Hole) 曲面生成器
# 依據用戶實際掃描樣品參數：
#   開口 228.8 nm / 底部 183.5 nm / 深度 125.8 nm
# ============================================================

def trapezoid_hole_creator(centers, half_opens_px, half_bots_px,
                           depths_nm, size):
    """
    向量化生成梯形孔截頭方錐形凹坑（square frustum pit）。

    截面形狀（Chebyshev 距離 d 從孔中心起）：
      d ≤ half_bot  : 孔底（均勻 -depth nm）
      half_bot < d ≤ half_open : 斜壁（線性從 -depth 到 0）
      d > half_open : 平坦表面（0 nm）

    Args:
        centers      : list of (row, col) 孔中心座標 (px，可為浮點)
        half_opens_px: list of 開口半寬 (px)
        half_bots_px : list of 底部半寬 (px)
        depths_nm    : list of 深度 (nm，正數)
        size         : 影像邊長 (px)
    Returns:
        surface : ndarray (size, size)，單位 nm，0=平面，負值=孔洞
    """
    rr, cc = np.indices((size, size))
    surface = np.zeros((size, size), dtype=np.float64)

    for (r0, c0), h_open, h_bot, depth in zip(
            centers, half_opens_px, half_bots_px, depths_nm):

        # Chebyshev 距離（正方形截面）
        d = np.maximum(np.abs(rr - r0), np.abs(cc - c0))

        z = np.zeros((size, size))

        # 孔底區域
        bottom_mask = d <= h_bot
        z[bottom_mask] = -depth

        # 斜壁區域
        wall_mask = (d > h_bot) & (d <= h_open)
        if (h_open - h_bot) > 1e-6:
            t = (d[wall_mask] - h_bot) / (h_open - h_bot)
            z[wall_mask] = -depth * (1.0 - t)

        # 多孔重疊時取最深值
        surface = np.minimum(surface, z)

    return surface


def trapezoid_randomizer(px_nm=SURFACE_SCALE_NM, img_size=128,
                         max_attempts=2000):
    """
    生成梯形孔隨機參數（±TRAP_VARIATION 變異，1–3 個孔，不重疊）。

    Returns:
        centers      : list of (row, col)
        half_opens_px: list of 開口半寬 (px)
        half_bots_px : list of 底部半寬 (px)
        depths_nm    : list of 深度 (nm)
    """
    n_holes = random.randint(1, 3)
    centers, half_opens_px, half_bots_px, depths_nm = [], [], [], []

    for _ in range(n_holes):
        v = TRAP_VARIATION
        open_nm  = random.uniform(TRAP_OPEN_NM  * (1 - v), TRAP_OPEN_NM  * (1 + v))
        bot_nm   = random.uniform(TRAP_BOT_NM   * (1 - v), TRAP_BOT_NM   * (1 + v))
        depth_nm = random.uniform(TRAP_DEPTH_NM * (1 - v), TRAP_DEPTH_NM * (1 + v))

        # 確保 bottom ≤ open（物理限制）
        bot_nm = min(bot_nm, open_nm * 0.95)

        h_open = (open_nm / px_nm) / 2.0   # 半寬 (px)
        h_bot  = (bot_nm  / px_nm) / 2.0

        margin   = h_open + 3
        min_pos  = margin
        max_pos  = img_size - margin

        if max_pos < min_pos:
            break   # 影像太小，放不下

        placed = False
        for _ in range(max_attempts):
            r = random.uniform(min_pos, max_pos)
            c = random.uniform(min_pos, max_pos)

            overlap = any(
                max(abs(r - pr), abs(c - pc)) < (h_open + po) * 1.05
                for (pr, pc), po in zip(centers, half_opens_px)
            )
            if not overlap:
                centers.append((r, c))
                half_opens_px.append(h_open)
                half_bots_px.append(h_bot)
                depths_nm.append(depth_nm)
                placed = True
                break

        if not placed:
            break   # 放不下更多孔

    return centers, half_opens_px, half_bots_px, depths_nm


def load_and_prepare_tip(mat_path, key='tip', target_size=TARGET_TIP_SIZE):
    """
    載入 AFM 探針估計結果，修正方向與尺寸。

    FIX A:
      - 找出最大值位置（探針頂端）並移至中心（shift）
      - 強制 max = 0
      - 套用徑向平均，使探針具有旋轉對稱性

    FIX B:
      - Resize 至 target_size × target_size（預設 55×55）
      - 確保 resize 後 max 仍為 0
    """
    raw = io.loadmat(mat_path)[key].astype(np.float64)

    # FIX A-1：將最大值移到中心
    max_idx = np.unravel_index(raw.argmax(), raw.shape)
    cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
    shifted = ndimage.shift(raw, [cy - max_idx[0], cx - max_idx[1]], mode='nearest')
    shifted -= shifted.max()  # 確保 max = 0

    # FIX A-2：徑向平均 → 強制旋轉對稱
    ny, nx = shifted.shape
    cy2, cx2 = ny // 2, nx // 2
    y_idx, x_idx = np.indices((ny, nx))
    r_map = np.round(np.sqrt((y_idx - cy2)**2 + (x_idx - cx2)**2)).astype(int)
    radially_averaged = np.zeros_like(shifted)
    for ri in range(r_map.max() + 1):
        mask = r_map == ri
        if mask.any():
            radially_averaged[mask] = shifted[mask].mean()
    radially_averaged -= radially_averaged.max()  # 再確保 max = 0

    # FIX B：Resize 至 target_size × target_size
    zoom_factor = target_size / radially_averaged.shape[0]
    tip_out = ndimage.zoom(radially_averaged, zoom_factor, order=3)
    tip_out -= tip_out.max()  # 確保插值後 max 仍為 0

    print(f"  Tip loaded from: {mat_path}")
    print(f"  Raw shape: {raw.shape}  →  Prepared shape: {tip_out.shape}")
    print(f"  Raw max idx: {max_idx}  →  center: ({cy}, {cx})")
    print(f"  Tip value range: [{tip_out.min():.3f}, {tip_out.max():.3f}] nm")
    return tip_out


def load_tip_for_training(mat_path, training_px_nm=SURFACE_SCALE_NM, max_tip_px=21):
    """
    tip.mat → 重採樣至訓練影像像素尺度 training_px_nm，供 grey_dilation 使用。

    訓練影像尺度 (SURFACE_SCALE_NM=39.1 nm/px) 通常與 AFM 掃描尺度不同
    （典型掃描：5000nm/256px ≈ 19.5 nm/px）。需先縮放使物理尺寸一致，
    再裁剪至 max_tip_px 以兼顧效率。

    Steps:
      1. 頂端置中 + max=0
      2. 徑向平均 → 旋轉對稱
      3. zoom = tip_px_nm / training_px_nm（物理尺寸不變，像素數縮放）
      4. 裁剪至 max_tip_px × max_tip_px（奇數）
    """
    mat = io.loadmat(mat_path)

    # 讀取探針陣列（支援 'tip' 與 'tip_estimated' 兩種鍵名）
    raw = None
    for k in ('tip', 'tip_estimated'):
        if k in mat:
            raw = np.array(mat[k]).astype(np.float64)
            break
    if raw is None:
        user_keys = [k for k in mat if not k.startswith('_')]
        raise KeyError(f"tip.mat 中找不到 'tip'/'tip_estimated'，現有鍵：{user_keys}")

    # 讀取來源像素尺度（afm_gui.py 儲存 tip 時會附帶 px_nm）
    if 'px_nm' in mat:
        tip_px_nm = float(np.array(mat['px_nm']).flat[0])
    else:
        tip_px_nm = 5000.0 / 256.0   # 預設：5000nm 掃描 / 256px ≈ 19.53 nm/px
        print(f"  [tip] px_nm 未記錄，假設預設值 {tip_px_nm:.2f} nm/px")

    # Step 1: 頂端置中
    max_idx = np.unravel_index(raw.argmax(), raw.shape)
    cy, cx  = raw.shape[0] // 2, raw.shape[1] // 2
    centered = ndimage.shift(raw, [cy - max_idx[0], cx - max_idx[1]], mode='nearest')
    centered -= centered.max()

    # Step 2: 徑向平均 → 旋轉對稱
    ny, nx = centered.shape
    cy2, cx2 = ny // 2, nx // 2
    y_idx, x_idx = np.indices((ny, nx))
    r_map = np.round(np.sqrt((y_idx - cy2)**2 + (x_idx - cx2)**2)).astype(int)
    radial = np.zeros_like(centered)
    for ri in range(r_map.max() + 1):
        mask = r_map == ri
        if mask.any():
            radial[mask] = centered[mask].mean()
    radial -= radial.max()

    # Step 3: 縮放至訓練像素尺度
    #   zoom_factor = tip_px_nm / training_px_nm
    #   物理尺寸保持不變；每 px 代表更大物理距離時陣列縮小
    zoom_factor = tip_px_nm / training_px_nm
    scaled = ndimage.zoom(radial, zoom_factor, order=3)
    scaled -= scaled.max()

    # Step 4: 裁剪至 max_tip_px（保持奇數、以頂端為中心）
    h, w = scaled.shape
    if h > max_tip_px or w > max_tip_px:
        cy_s = scaled.shape[0] // 2
        cx_s = scaled.shape[1] // 2
        half  = max_tip_px // 2
        scaled = scaled[cy_s - half : cy_s + half + 1,
                        cx_s - half : cx_s + half + 1]
        scaled -= scaled.max()

    # 確保奇數尺寸（ndimage.zoom 可能產生偶數）
    if scaled.shape[0] % 2 == 0:
        scaled = scaled[:-1, :]
    if scaled.shape[1] % 2 == 0:
        scaled = scaled[:, :-1]

    print(f"  [tip] 來源: {mat_path}  (tip_px_nm={tip_px_nm:.2f})")
    print(f"  重採樣: {raw.shape} → {scaled.shape} @ {training_px_nm} nm/px")
    print(f"  深度範圍: [{scaled.min():.1f}, {scaled.max():.1f}] nm")
    return scaled


# Custom callback for training progress bar and metrics display
class TrainingProgressCallback(Callback):
    """Custom callback to display training progress bar and metrics per epoch."""

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.epoch_times = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        print("\n" + "="*70)
        print(f"開始訓練 - 總 Epochs: {self.epochs}")
        print("="*70 + "\n")
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)

        # [Fix 8] 移除無意義的 "100 - MAE" 準確率，改顯示 val/train loss 比值
        # 比值越接近 1.0 表示模型泛化越好（無過擬合）
        loss_ratio = (val_loss / loss) if loss > 0 else float('nan')

        # 計算預計剩餘時間
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.epochs - self.current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        # 格式化輸出
        print(f"\rEpoch {self.current_epoch}/{self.epochs} | "
              f"Loss: {loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Val/Train Ratio: {loss_ratio:.4f} | "
              f"Time: {epoch_time:.2f}s | ETA: {eta_seconds//60:.0f}m {eta_seconds%60:.0f}s")

        # 每 10 個 epoch 顯示進度條
        if self.current_epoch % 10 == 0 or self.current_epoch == self.epochs:
            progress = (self.current_epoch / self.epochs) * 100
            bar_length = 30
            filled = int(bar_length * self.current_epoch / self.epochs)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"Progress: [{bar}] {progress:.1f}%")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n" + "="*70)
        print(f"訓練完成！總耗時: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.2f}s")
        print("="*70 + "\n")


# 數據生成進度回調
class DataGenerationCallback:
    """用於顯示數據生成進度。"""

    def __init__(self, total, desc="Generating"):
        self.pbar = tqdm(total=total, desc=desc, unit="samples",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def update(self, n=1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()

# 檢查並設置 GPU
print("\n" + "="*70)
print("TensorFlow GPU 檢查")
print("="*70)
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 可用: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA 建立: {tf.test.is_built_with_cuda()}")

if tf.config.list_physical_devices('GPU'):
    print("✓ GPU 已啟用，將使用 GPU 進行訓練")
    # 設置 GPU 記憶體增長，避免一次性佔用所有顯存
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠ GPU 不可用，將使用 CPU 進行訓練")
print("="*70 + "\n")

# ============================================================
# 孔洞樣品模擬（全部為凹洞，與真實 AFM 掃描一致）
# 移除：球形/方形/圓柱突起粒子（物理行為與孔洞完全相反，不應混訓）
# 保留：梯形孔（trapezoid_hole）+ 新增圓柱孔（cylinder_hole）
# ============================================================
size = 128

# ============================================================
# Simulate cylindrical holes (圓柱形孔洞)
#   圓形截面，垂直側壁，平底
#   直徑：8–20 px → 40–100 nm（5 nm/px）
#   深度：16–25 px → 80–125 nm
# ============================================================

def cylinder_hole_randomizer(px_nm=SURFACE_SCALE_NM, img_size=128,
                              max_attempts=5000):
    """
    生成不重疊的圓柱孔隨機參數（1–3 個孔）。
    Returns:
        centers_px  : list of (row, col) 孔中心 (px)
        radii_px    : list of 半徑 (px)
        depths_nm   : list of 深度 (nm, 正數)
    """
    n_holes = random.randint(1, 6)       # 最多 6 個，模擬密集孔陣
    centers_px, radii_px, depths_nm = [], [], []

    for _ in range(n_holes):
        # ── 尺度多樣性（at 39.1 nm/px）：混合小孔與大孔豐富資料集 ──────
        #   70% 小孔 (1.5–8px = 59–313nm)：訓練極小/一般孔的去卷積
        #   30% 大孔 (8–24px = 313–938nm)：訓練大尺度孔洞、提升泛化
        # 大孔讓模型不會只學到「小圓點」的偏誤，增強尺度魯棒性
        if random.random() < 0.30:
            r_px = random.uniform(8.0, 24.0)     # 大孔 313–938 nm
        else:
            r_px = random.uniform(1.5, 8.0)      # 小孔 59–313 nm
        depth   = random.uniform(60.0, 145.0)   # 深度 60–145 nm（加寬變異）
        margin  = r_px + 3
        min_pos = margin
        max_pos = img_size - margin
        if max_pos < min_pos:
            break
        placed  = False
        for _ in range(max_attempts):
            r = random.uniform(min_pos, max_pos)
            c = random.uniform(min_pos, max_pos)
            overlap = any(
                np.sqrt((r - pr)**2 + (c - pc)**2) < (r_px + er) * 1.1
                for (pr, pc), er in zip(centers_px, radii_px)
            ) if centers_px else False
            if not overlap:
                centers_px.append((r, c))
                radii_px.append(r_px)
                depths_nm.append(depth)
                placed = True
                break
        if not placed:
            break

    return centers_px, radii_px, depths_nm


def cylinder_hole_creator(centers_px, radii_px, depths_nm, size):
    """
    向量化生成圓柱孔：
      d ≤ r  : 孔底（均勻 -depth nm）
      d > r  : 平坦表面（0 nm）
    """
    rr, cc = np.indices((size, size))
    surface = np.zeros((size, size), dtype=np.float64)
    for (r0, c0), r_px, depth in zip(centers_px, radii_px, depths_nm):
        d = np.sqrt((rr - r0)**2 + (cc - c0)**2)
        z = np.zeros((size, size))
        z[d <= r_px] = -depth
        surface = np.minimum(surface, z)
    return surface


# ============================================================
# Simulate star-shaped holes (星形孔洞)
#   n 芒星形截面，半徑隨角度在內/外半徑間振盪：
#     r_bound(θ) = r_mid + amp · cos(n_points · (θ − θ0))
#   提供「非凸 + 含尖角」幾何，強迫模型學習各方向的去卷積，
#   不再只擅長對稱圓孔 → 提升形狀魯棒性
# ============================================================

def star_hole_randomizer(px_nm=SURFACE_SCALE_NM, img_size=128,
                          max_attempts=5000):
    """
    生成不重疊的星形孔隨機參數（1–3 個孔）。
    Returns:
        centers_px : list of (row, col)
        r_out_px   : list of 外接半徑 (px)
        r_in_px    : list of 內接半徑 (px)
        n_points   : list of 星芒數 (int)
        angles     : list of 旋轉角 θ0 (rad)
        depths_nm  : list of 深度 (nm, 正數)
    """
    n_holes = random.randint(1, 3)
    centers_px, r_out_px, r_in_px = [], [], []
    n_points, angles, depths_nm = [], [], []

    for _ in range(n_holes):
        r_out = random.uniform(8.0, 20.0)        # 外接半徑 313–782 nm
        ratio = random.uniform(0.40, 0.70)       # 內/外半徑比（越小越尖）
        r_in  = r_out * ratio
        npts  = random.choice([4, 5, 6])         # 星芒數
        ang0  = random.uniform(0, 2 * math.pi)   # 隨機旋轉，避免方向偏誤
        depth = random.uniform(60.0, 145.0)

        margin  = r_out + 3
        min_pos = margin
        max_pos = img_size - margin
        if max_pos < min_pos:
            break
        placed = False
        for _ in range(max_attempts):
            r = random.uniform(min_pos, max_pos)
            c = random.uniform(min_pos, max_pos)
            # 以外接半徑做保守的不重疊判斷
            overlap = any(
                np.sqrt((r - pr)**2 + (c - pc)**2) < (r_out + pro) * 1.1
                for (pr, pc), pro in zip(centers_px, r_out_px)
            ) if centers_px else False
            if not overlap:
                centers_px.append((r, c))
                r_out_px.append(r_out)
                r_in_px.append(r_in)
                n_points.append(npts)
                angles.append(ang0)
                depths_nm.append(depth)
                placed = True
                break
        if not placed:
            break

    return centers_px, r_out_px, r_in_px, n_points, angles, depths_nm


def star_hole_creator(centers_px, r_out_px, r_in_px, n_points,
                      angles, depths_nm, size):
    """
    向量化生成星形孔（平底、垂直側壁）：
      r ≤ r_bound(θ) : 孔底（均勻 -depth nm）
      其餘            : 平坦表面（0 nm）
    其中 r_bound(θ) = r_mid + amp · cos(n_points · (θ − θ0))
    """
    rr, cc = np.indices((size, size))
    surface = np.zeros((size, size), dtype=np.float64)
    for (r0, c0), ro, ri, npts, ang0, depth in zip(
            centers_px, r_out_px, r_in_px, n_points, angles, depths_nm):
        dy = rr - r0
        dx = cc - c0
        r     = np.sqrt(dy**2 + dx**2)
        theta = np.arctan2(dy, dx)
        r_mid = (ro + ri) / 2.0
        amp   = (ro - ri) / 2.0
        r_bound = r_mid + amp * np.cos(npts * (theta - ang0))
        z = np.zeros((size, size))
        z[r <= r_bound] = -depth
        surface = np.minimum(surface, z)
    return surface


# ============================================================
# 真實掃描 artifact 模擬（只加到 X/input，不加到 GT）
# ============================================================

def add_scan_line_artifacts(img, n_events=None, max_band_px=4,
                             amplitude_nm=20.0):
    """
    模擬 AFM 逐行掃描的水平條紋 Z 漂移 artifact。
    每個 event：隨機選 row，加上隨機正/負偏移，寬度 1–max_band_px 行。

    振幅從 8nm 提升至 20nm，與真實 AFM 掃描的條紋強度一致。
    50% 機率完全不加（讓模型同時看過有/無 artifact 的樣本）。
    """
    if random.random() < 0.5:
        return img.copy()
    result = img.copy()
    h = img.shape[0]
    if n_events is None:
        n_events = random.randint(1, 8)
    for _ in range(n_events):
        row0  = random.randint(0, h - 1)
        width = random.randint(1, max_band_px)
        amp   = random.gauss(0, amplitude_nm)
        result[row0:min(row0 + width, h), :] += amp
    return result


def add_gaussian_noise(img, sigma_nm=None):
    """
    加入高斯隨機雜訊，模擬 AFM 熱雜訊 + 電子雜訊。
    sigma_nm 預設從 0.5–2.5 nm 隨機選取。
    """
    if sigma_nm is None:
        sigma_nm = random.uniform(0.5, 2.5)
    return img + np.random.normal(0, sigma_nm, img.shape).astype(np.float32)


# ---- 圓柱孔 N_CYL=1800（圓形樣品主要訓練形狀，佔比最大）-----------
N_CYL = 1800
list_cyl_hole_images = []

print(f"Generating cylinder holes (r=1.5–24px={1.5*SURFACE_SCALE_NM:.0f}–{24*SURFACE_SCALE_NM:.0f}nm 小+大混合, "
      f"depth=60–145nm, 1–6 holes/img, N={N_CYL})...")
for i in tqdm(range(N_CYL), desc="Cylinder Holes", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    ctrs, rads, deps = cylinder_hole_randomizer(px_nm=SURFACE_SCALE_NM, img_size=size)
    if not ctrs:                        # 極少情況放不下任何孔，放一個在中心
        ctrs  = [(size//2, size//2)]
        rads  = [5.0]
        deps  = [100.0]
    surf = cylinder_hole_creator(ctrs, rads, deps, size)
    list_cyl_hole_images.append(surf)

true_cyl_hole_stack = np.stack(list_cyl_hole_images, axis=0)
np.save(f'{OUTPUT_DIR}/true_cyl_hole_stack.npy', true_cyl_hole_stack)

# Preview
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, idx in zip(axes, [0, 1, 2]):
    im = ax.imshow(true_cyl_hole_stack[idx], cmap='viridis')
    ax.set_title(f"Cylinder Hole GT #{idx}  "
                 f"(min={true_cyl_hole_stack[idx].min():.1f} nm)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='nm')
plt.tight_layout()
save_plot('cylinder_hole_preview.png')


# ============================================================
# Simulate trapezoidal holes  ← 用戶真實樣品幾何
#   開口 228.8 nm / 底部 183.5 nm / 深度 125.8 nm
#   ±20% 隨機變異；1-3 個孔/張；單位直接為 nm
# ============================================================

list_trap_gndtruth_image = []

# 梯形孔：方形 Chebyshev 截面對圓形樣品會引入角狀歧義，適度減量
N_TRAP = 600

print(f"Generating trapezoidal holes "
      f"(open={TRAP_OPEN_NM}nm, bot={TRAP_BOT_NM}nm, "
      f"depth={TRAP_DEPTH_NM}nm, ±{int(TRAP_VARIATION*100)}%, N={N_TRAP})...")
for i in tqdm(range(N_TRAP), desc="Trapezoid", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    centers, h_opens, h_bots, depths = trapezoid_randomizer(
        px_nm=SURFACE_SCALE_NM, img_size=size)
    true_trap = trapezoid_hole_creator(
        centers, h_opens, h_bots, depths, size)
    list_trap_gndtruth_image.append(true_trap)

true_trap_surf_stack = np.stack(list_trap_gndtruth_image, axis=0)
np.save(f'{OUTPUT_DIR}/true_trap_stack.npy', true_trap_surf_stack)

# 視覺化檢查（顯示前 2 張：ground-truth）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, idx in zip(axes, [0, 1, 2]):
    if idx < len(true_trap_surf_stack):
        im = ax.imshow(true_trap_surf_stack[idx], cmap='viridis')
        ax.set_title(f"Trapezoid GT #{idx}  "
                     f"(min={true_trap_surf_stack[idx].min():.1f} nm)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='nm')
plt.tight_layout()
save_plot('trapezoid_preview.png')


# 星形孔洞（star_hole_randomizer / star_hole_creator）已定義但不加入訓練：
# 樣品確認為圓形/平滑孔，星形資料會造成模型對圓形輸入「猜測」尖角存在（角狀歧義）。
# grey_dilation 是多對一映射：圓/方/星膨脹後外觀相似，混合訓練增加歧義。
# 若未來樣品包含非凸形狀，取消下方注釋即可恢復。


# ============================================================
# 套用 Grey Dilation（模擬 AFM 掃描，凹洞版本）
# Grey dilation 對孔洞的效果：
#   dilated[i,j] = max_{u,v}{ surface[i-u,j-v] + tip[u,v] }
#   孔洞（負值）+ tip（0 到負值）→ 孔洞在 dilated 中變淺/變窄
#   這正確模擬 AFM 探針無法完全進入孔洞的物理現象
# ============================================================

# 優先載入真實 tip.mat（探針形狀與實際掃描一致，去卷積更準確）
# 若不存在則退回合成拋物線探針
TIP_MAT_CANDIDATES = [
    'tip.mat/tip_estimated_corrected.mat',   # 校正後（優先）
    'tip.mat/tip_estimated.mat',             # 未校正
    'tip.mat',                               # 通用名稱
]
tip_shape  = None
tip_source = 'synthetic_paraboloid'
for _tp in TIP_MAT_CANDIDATES:
    if os.path.exists(_tp):
        try:
            # max_tip_px=5 → 5×5 px = ±2 px = ±78 nm 有效半徑；
            # 比舊版 21 px 縮小，使訓練探針（半寬 78 nm）接近真實 ROC=46.9 nm
            # 的有效影響範圍（sqrt(2×46.9×125)/39.1 ≈ 2.77 px），
            # 防止小孔在 dilation 時被完全填平導致模型過度去卷積。
            tip_shape  = load_tip_for_training(_tp, max_tip_px=5)
            tip_source = _tp
            print(f"  [探針] 使用真實探針：{_tp}  "
                  f"({tip_shape.shape[0]}×{tip_shape.shape[1]} px)")
            break
        except Exception as _e:
            print(f"  [警告] 載入 {_tp} 失敗：{_e}，嘗試下一候選")

if tip_shape is None:
    print("建立訓練用合成探針（拋物線，R=73nm）...")
    tip_shape  = make_training_tip()
    tip_source = 'synthetic_paraboloid'

_tip_label = (os.path.basename(tip_source) if tip_source != 'synthetic_paraboloid'
              else f'Paraboloid R={TIP_RADIUS_NM}nm')
plt.figure()
plt.imshow(tip_shape, cmap='viridis')
plt.colorbar()
plt.title(f"Training Tip — {_tip_label}  {tip_shape.shape[0]}×{tip_shape.shape[1]} px")
save_plot('tip_shape.png')

# 載入兩種孔洞 ground-truth（已移除星形：對圓形樣品會增加角狀歧義）
true_trap_surf_stack     = np.load(f'{OUTPUT_DIR}/true_trap_stack.npy').astype('float32')
true_cyl_hole_surf_stack = np.load(f'{OUTPUT_DIR}/true_cyl_hole_stack.npy').astype('float32')

# ── 梯形孔 dilation ──
list_trap_dilated_image = []
print(f"Dilation: {len(true_trap_surf_stack)} trapezoid images...")
for i in tqdm(range(len(true_trap_surf_stack)), desc="Dilation(Trap)", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    surf = true_trap_surf_stack[i, :, :]
    dilated = scipy.ndimage.grey_dilation(surf, structure=tip_shape)
    dilated = add_scan_line_artifacts(dilated)
    dilated = add_gaussian_noise(dilated)
    list_trap_dilated_image.append(dilated)
dilated_trap_surf_stack = np.stack(list_trap_dilated_image, axis=0)
np.save(f'{OUTPUT_DIR}/dilated_trap_stack.npy', dilated_trap_surf_stack)

# ── 圓柱孔 dilation ──
list_cyl_hole_dilated_image = []
print(f"Dilation: {len(true_cyl_hole_surf_stack)} cylinder images...")
for i in tqdm(range(len(true_cyl_hole_surf_stack)), desc="Dilation(Cyl)", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    surf = true_cyl_hole_surf_stack[i, :, :]
    dilated = scipy.ndimage.grey_dilation(surf, structure=tip_shape)
    dilated = add_scan_line_artifacts(dilated)
    dilated = add_gaussian_noise(dilated)
    list_cyl_hole_dilated_image.append(dilated)
dilated_cyl_hole_surf_stack = np.stack(list_cyl_hole_dilated_image, axis=0)
np.save(f'{OUTPUT_DIR}/dilated_cyl_hole_stack.npy', dilated_cyl_hole_surf_stack)

# 快速視覺驗證（確認孔洞在 dilation 後變淺/變窄，符合 AFM 物理）
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for col in range(4):
    axes[0, col].imshow(true_trap_surf_stack[col], cmap='viridis',
                        vmin=-160, vmax=10)
    axes[0, col].set_title(f'Trap GT #{col}')
    axes[0, col].axis('off')
    axes[1, col].imshow(dilated_trap_surf_stack[col], cmap='viridis',
                        vmin=-160, vmax=10)
    axes[1, col].set_title(f'Trap Dilated #{col}')
    axes[1, col].axis('off')
plt.suptitle('梯形孔：Ground-Truth vs Dilated（孔洞應變淺/變窄）', fontsize=11)
plt.tight_layout()
save_plot('trap_dilation_check.png')


# ============================================================
# Load images and prepare training/testing datasets
# ============================================================
# 訓練資料全部為孔洞（凹洞）：
#   X1 / y1 : 梯形孔 (trapezoid_hole)  N_TRAP=600  張
#   X2 / y2 : 圓柱孔 (cylinder_hole)   N_CYL=1800  張（小+大混合尺度，圓形樣品主形狀）
#   共 2400 張，train:test = 8:2 → train 1920 / test 480
#   X（dilated）含水平條紋 artifact（50%）+ 高斯雜訊，GT 不含
#   星形孔已移除：grey dilation 使圓/方/星膨脹後形狀相近，
#   訓練含星形會令模型對圓形輸入猜測尖角（角狀歧義）。

dilated_images_path_trap     = f'{OUTPUT_DIR}/dilated_trap_stack.npy'
true_images_path_trap        = f'{OUTPUT_DIR}/true_trap_stack.npy'
dilated_images_path_cyl_hole = f'{OUTPUT_DIR}/dilated_cyl_hole_stack.npy'
true_images_path_cyl_hole    = f'{OUTPUT_DIR}/true_cyl_hole_stack.npy'

X1 = np.load(dilated_images_path_trap).astype('float32')
y1 = np.load(true_images_path_trap).astype('float32')
X2 = np.load(dilated_images_path_cyl_hole).astype('float32')
y2 = np.load(true_images_path_cyl_hole).astype('float32')

# Split（兩類各自 8:2，避免某類別只落在 train 或 test）
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_merged_train = np.concatenate((X_1_train, X_2_train), axis=0)
y_merged_train = np.concatenate((y_1_train, y_2_train), axis=0)

X_merged_train, y_merged_train = shuffle(X_merged_train, y_merged_train)

X_merged_test = np.concatenate((X_1_test, X_2_test), axis=0)
y_merged_test = np.concatenate((y_1_test, y_2_test), axis=0)

print("Training data loaded.",
      f"\n  ✓ Trapezoid holes : {X1.shape[0]} imgs（N_TRAP={N_TRAP}）",
      f"\n  ✓ Cylinder  holes : {X2.shape[0]} imgs（N_CYL={N_CYL}，小+大混合尺度）",
      f"\n  ✗ Star holes      : 已移除（對圓形樣品增加角狀歧義）",
      f"\n  ✗ Sphere/Cubic/Poly: 已移除（突起資料，物理行為相反）",
      f"\n  Train total: {X_merged_train.shape[0]}",
      f"\n  Test  total: {X_merged_test.shape[0]}",
      f"\n  Resolution : {X_merged_train.shape[1:3]}")


# ============================================================
# Preprocessing
# ============================================================

# Height correction for tip-convoluted images (X)
def height_correcter_X(image_data, image_size):
    processed_images_l = []
    for image in tqdm(image_data, desc="Preprocessing X", unit="img", leave=False):
        processed_image = image + np.full((128, 128), 0.0)  # Background subtraction
        processed_images_l.append(processed_image.reshape(image_size[0], image_size[1], 1))
    processed_images_a = np.stack(processed_images_l, axis=0)
    return processed_images_a

# Preprocessing for ground-truth images (y) — reshape only
def preprocessing_Y(image_data, image_size):
    processed_images_l = []
    for image in tqdm(image_data, desc="Preprocessing Y", unit="img", leave=False):
        processed_image = image
        processed_images_l.append(processed_image.reshape(image_size[0], image_size[1], 1))
    processed_images_a = np.stack(processed_images_l, axis=0)
    return processed_images_a

# Apply the preprocessing functions
target_size = (128, 128)
# [Fix 7] 移除未使用的 nb_classes = 2

X_processed_train = height_correcter_X(X_merged_train, target_size)
X_processed_test  = height_correcter_X(X_merged_test, target_size)
y_processed_train = preprocessing_Y(y_merged_train, target_size)
y_processed_test  = preprocessing_Y(y_merged_test, target_size)

# ---- 全域正規化 → [0, 1]（解決模型輸出全零問題）-----------------
print(f"\n{'='*60}")
print(f"  全域正規化  NORM_MIN={NORM_MIN} nm  NORM_MAX={NORM_MAX} nm")
print(f"  背景 0 nm → {(0-NORM_MIN)/(NORM_MAX-NORM_MIN):.3f}（不再是特殊零值）")
print(f"{'='*60}")
X_processed_train = normalize_data(X_processed_train)
X_processed_test  = normalize_data(X_processed_test)
y_processed_train = normalize_data(y_processed_train)
y_processed_test  = normalize_data(y_processed_test)
print(f"  X_train  : [{X_processed_train.min():.3f}, {X_processed_train.max():.3f}]")
print(f"  y_train  : [{y_processed_train.min():.3f}, {y_processed_train.max():.3f}]")
print(f"  X_test   : [{X_processed_test.min():.3f}, {X_processed_test.max():.3f}]")
print(f"  y_test   : [{y_processed_test.min():.3f}, {y_processed_test.max():.3f}]\n")

# Show the processed images with scale
fig = plt.figure(figsize=(10, 7))
rows = 3
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(X_processed_train[0].reshape(128, 128))
plt.axis('off')
plt.colorbar()
plt.title("Input (dilated) 0")

fig.add_subplot(rows, columns, 2)
plt.imshow(y_processed_train[0].reshape(128, 128))
plt.axis('off')
plt.colorbar()
plt.title("Label (true) 0")

fig.add_subplot(rows, columns, 3)
plt.imshow(X_processed_train[4].reshape(128, 128))
plt.axis('off')
plt.colorbar()
plt.title("Input (dilated) 4")

fig.add_subplot(rows, columns, 4)
plt.imshow(y_processed_train[4].reshape(128, 128))
plt.axis('off')
plt.colorbar()
plt.title("Label (true) 4")

fig.add_subplot(rows, columns, 5)
plt.imshow(X_processed_train[5].reshape(128, 128))
plt.axis('off')
plt.colorbar()
plt.title("Input (dilated) 5")

fig.add_subplot(rows, columns, 6)
plt.imshow(y_processed_train[5].reshape(128, 128))
plt.axis('off')
plt.colorbar()
plt.title("Label (true) 5")

plt.tight_layout()
save_plot('training_samples.png')


# ============================================================
# Neural Network Architecture — U-Net with Skip Connections
# ============================================================
# 原 Autoencoder 問題：4 次 MaxPool 將空間壓縮至 8×8（bottleneck
# 僅 2048 值），孔洞位置與形狀資訊大量丟失，導致預測邊緣鋸齒、
# 大小不一致。
#
# U-Net 解法：Skip Connection 繞過 bottleneck 直接傳遞精確空間
# 特徵，decoder 每級皆可參考對應解析度的 encoder 輸出。
#
#   Encoder  : 128→64→32→16 px（3×MaxPool），通道 32→64→128
#   Bottleneck: 16×16×256（保留 65536 值，原為 2048，32×改善）
#   Decoder  : 16→32→64→128 px（UpSampling + concat skip）
#   每卷積塊  : Conv3×3 → BatchNorm → ReLU（×2）
#   輸出      : Conv1×1 Sigmoid → [0,1]
#
#   背景 0 nm → 0.625（NORM_MIN=-175），模型無法偷懶全輸出常數
# ============================================================

print(X_processed_train.shape[1:])
ini = tf.keras.initializers.HeNormal(1)

def conv_block(x, filters):
    """標準 U-Net 卷積塊：Conv→BN→ReLU ×2"""
    x = Convolution2D(filters, (3, 3), padding='same',
                      kernel_initializer=ini)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Convolution2D(filters, (3, 3), padding='same',
                      kernel_initializer=ini)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

inp = Input(shape=X_processed_train.shape[1:])

# ── Encoder ──────────────────────────────────────────────────
c1 = conv_block(inp, 32)                      # 128×128×32
p1 = MaxPooling2D((2, 2))(c1)                # 64×64

c2 = conv_block(p1, 64)                       # 64×64×64
p2 = MaxPooling2D((2, 2))(c2)                # 32×32

c3 = conv_block(p2, 128)                      # 32×32×128
p3 = MaxPooling2D((2, 2))(c3)                # 16×16

# ── Bottleneck ───────────────────────────────────────────────
c4 = conv_block(p3, 256)                      # 16×16×256

# ── Decoder（UpSampling → concat skip → conv_block）─────────
u5 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(c4)
u5 = Concatenate()([u5, c3])                  # 32×32×256
c5 = conv_block(u5, 128)

u6 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(c5)
u6 = Concatenate()([u6, c2])                  # 64×64×128
c6 = conv_block(u6, 64)

u7 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(c6)
u7 = Concatenate()([u7, c1])                  # 128×128×64
c7 = conv_block(u7, 32)

# ── Output ───────────────────────────────────────────────────
output = Convolution2D(1, (1, 1), activation='sigmoid',
                       padding='same', kernel_initializer=ini)(c7)

autoencoder = Model(inputs=inp, outputs=output)
autoencoder.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
autoencoder.compile(optimizer=opt, loss='MAE')


# ============================================================
# Training
# ============================================================

EPOCHS = 3000      # EarlyStopping 會提前結束，不一定跑滿
BATCH_SIZE = 32

progress_callback = TrainingProgressCallback(epochs=EPOCHS)

# ReduceLROnPlateau：val_loss 停滯 100 epoch → lr ×0.5（最低 1e-6）
# EarlyStopping：val_loss 停滯 400 epoch → 停止並還原最佳權重
reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=100,
    min_lr=1e-6, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=400,
    restore_best_weights=True, verbose=1)

print(f"\n訓練參數:")
print(f"  Epochs (max):       {EPOCHS}")
print(f"  Batch Size:         {BATCH_SIZE}")
print(f"  Training Samples:   {X_processed_train.shape[0]}")
print(f"  Validation Samples: {X_processed_test.shape[0]}")
print(f"  Input Shape:        {X_processed_train.shape[1:]}\n")

training_data = autoencoder.fit(
    X_processed_train, y_processed_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_processed_test, y_processed_test),
    callbacks=[progress_callback, reduce_lr, early_stop],
    verbose=0  # 使用自定義回調代替默認輸出
)

# ============================================================
# Save the trained model
# [Fix 6] 統一使用 .keras 格式存檔（移除 .h5 不一致問題）
# ============================================================

MODEL_PATH   = os.path.join(WEIGHTS_DIR, 'AFM_MAE_autoencoder.keras')
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'AFM_MAE_autoencoder.weights.h5')

autoencoder.save(MODEL_PATH)
autoencoder.save_weights(WEIGHTS_PATH)

np.save(os.path.join(RUN_DIR, 'train_history.npy'), training_data.history)
history = np.load(os.path.join(RUN_DIR, 'train_history.npy'), allow_pickle=True).item()

print(f'Model  → {MODEL_PATH}')
print(f'Weights→ {WEIGHTS_PATH}')
print('Model and weights saved.\n')

# Visualize the loss curve
plt.figure(figsize=(8, 4))
plt.plot(training_data.history['loss'], label='Train Loss')
plt.plot(training_data.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.ylabel('Loss (MAE)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
save_plot('loss_curve.png')


# ============================================================
# Prediction and Evaluation
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    計算評估指標。
    [Fix 9] SSIM 改用 skimage.metrics.structural_similarity 正確計算
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    # Mean Squared Error
    mse = np.mean((y_true - y_pred)**2)
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    # Peak Signal-to-Noise Ratio（以訊號動態範圍為基準，避免孔洞資料 max=0 導致 -inf）
    data_range_psnr = float(y_true.max() - y_true.min())
    if rmse != 0 and data_range_psnr > 0:
        psnr = 20 * np.log10(data_range_psnr / rmse)
    elif rmse == 0:
        psnr = float('inf')
    else:
        psnr = float('nan')
    # Structural Similarity Index — [Fix 9] 使用正確的 skimage SSIM
    y_true_sq = y_true.squeeze()  # (N, H, W)
    y_pred_sq = y_pred.squeeze()
    data_range = float(y_true_sq.max() - y_true_sq.min())
    ssim_vals = [
        ssim_func(y_true_sq[k], y_pred_sq[k], data_range=data_range)
        for k in range(len(y_true_sq))
    ]
    ssim = float(np.mean(ssim_vals))

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'PSNR': psnr, 'SSIM': ssim}


def get_decoded_imgs(input_imgs, filepath, nb_channels=1):
    print("\n載入模型進行預測...")
    model = load_model(filepath)
    print(f"預測 {input_imgs.shape[0]} 張圖片...")
    decoded_imgs = model.predict(input_imgs, verbose=1)
    decoded_imgs = decoded_imgs.reshape(
        input_imgs.shape[0], target_size[0], target_size[1], nb_channels)
    print("✓ 預測完成！\n")
    return decoded_imgs


# [Fix 6] 載入路徑統一改為 .keras
# [Fix 11] 保持為 numpy array，不轉換成 list
decoded_imgs = get_decoded_imgs(X_processed_test, MODEL_PATH)

# ---- 反正規化回 nm，指標與視覺化均以物理量 nm 呈現 --------------
decoded_imgs_nm   = denormalize_data(decoded_imgs)
y_processed_nm    = denormalize_data(y_processed_test)
X_processed_nm    = denormalize_data(X_processed_test)

# 計算並顯示預測評估指標
print("\n" + "="*70)
print("預測結果評估（單位：nm）")
print("="*70)
metrics = calculate_metrics(y_processed_nm, decoded_imgs_nm)
print(f"\n測試集評估指標:")
print(f"  MAE  (平均絕對誤差): {metrics['MAE']:.6f} nm")
print(f"  MSE  (均方誤差):      {metrics['MSE']:.6f} nm²")
print(f"  RMSE (均方根誤差):    {metrics['RMSE']:.6f} nm")
print(f"  PSNR (峰值信噪比):    {metrics['PSNR']:.2f} dB")
print(f"  SSIM (結構相似性):    {metrics['SSIM']:.4f}  [0=差, 1=完美]")

# 計算像素級準確率 (使用閾值，基於 nm 範圍)
threshold = 0.1  # 10% 誤差容限
diff = np.abs(y_processed_nm - decoded_imgs_nm)
max_val = np.max(np.abs(y_processed_nm))
accuracy = np.mean((diff / (max_val + 1e-10)) < threshold) * 100
print(f"  Pixel Accuracy (<10% error): {accuracy:.2f}%")
print("="*70 + "\n")


# ============================================================
# Visualize predictions (2×3 grid)
# ============================================================

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

for col in range(3):
    # Row 0: model output（反正規化後 nm）
    img_out = decoded_imgs_nm[col].reshape(target_size[0], target_size[1])
    ax[0, col].imshow(img_out, cmap='viridis')
    ax[0, col].set_title(f'Output {col + 1}  (nm)')
    ax[0, col].axis('off')
    # Row 1: model input（反正規化後 nm）
    img_in = X_processed_nm[col].reshape(target_size[0], target_size[1])
    ax[1, col].imshow(img_in, cmap='viridis')
    ax[1, col].set_title(f'Input {col + 1}  (nm)')
    ax[1, col].axis('off')

plt.tight_layout()
save_plot('predictions_grid.png')


# ============================================================
# Load saved model and make further predictions for visualization
# [Fix 6] 載入路徑改為 .keras
# ============================================================

model = load_model(MODEL_PATH)
decoded = model.predict(X_processed_test)

# 反正規化回 nm，剖面圖縱軸單位正確
decoded = denormalize_data(decoded)

# Squeeze channel dimension for visualization
X_test  = denormalize_data(X_processed_test).squeeze()
Y_test  = denormalize_data(y_processed_test).squeeze()
decoded = decoded.squeeze()


def show_triplet(idx, row_slice=None, col_slice=slice(None), label=""):
    """
    儲存單一樣本的 tip-convoluted / ground-truth / predicted 三張圖，
    並繪製指定切線的剖面圖。（不顯示視窗，直接存檔）
    """
    safe_label = label.replace(' ', '_')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [X_test[idx], Y_test[idx], decoded[idx]],
                                ['Tip-convoluted', 'Ground-truth', 'Predicted']):
        im = ax.imshow(data)
        ax.set_title(f'{title} [{label}]')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_plot(f'triplet_{safe_label}.png')

    if row_slice is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(decoded[idx][row_slice, col_slice], label='Predicted')
        plt.plot(X_test[idx][row_slice, col_slice],  label='Tip-convoluted')
        plt.plot(Y_test[idx][row_slice, col_slice],  label='Ground-truth')
        plt.legend()
        plt.title(f'Line Profile — Sample {idx} [{label}]')
        plt.xlabel('Pixel')
        plt.ylabel('Height (nm)')
        plt.tight_layout()
        save_plot(f'profile_{safe_label}.png')


# Sample 0 (row 65, full width)
show_triplet(0, row_slice=65, label="sample 0")

# 動態計算安全索引（避免超出測試集大小導致 IndexError）
n_test = len(decoded)
idx_mid = min(100, n_test - 1)   # ~測試集中段
idx_end = min(130, n_test - 1)   # ~測試集後段

# Sample mid (row 60, full width + zoomed 25:100)
show_triplet(idx_mid, row_slice=60, label=f"sample_{idx_mid}")
# Zoomed view
plt.figure(figsize=(8, 4))
plt.plot(decoded[idx_mid, 60, 25:100], label='Predicted')
plt.plot(X_test[idx_mid, 60, 25:100],  label='Tip-convoluted')
plt.plot(Y_test[idx_mid, 60, 25:100],  label='Ground-truth')
plt.legend()
plt.title(f'Line Profile — Sample {idx_mid}, row 60, col 25:100')
plt.tight_layout()
save_plot(f'profile_sample{idx_mid}_zoom.png')

# Sample end (row 20, full width + zoomed 30:90)
show_triplet(idx_end, row_slice=20, label=f"sample_{idx_end}")
plt.figure(figsize=(8, 4))
plt.plot(decoded[idx_end, 20, 30:90], label='Predicted')
plt.plot(X_test[idx_end, 20, 30:90],  label='Tip-convoluted')
plt.plot(Y_test[idx_end, 20, 30:90],  label='Ground-truth')
plt.legend()
plt.title(f'Line Profile — Sample {idx_end}, row 20, col 30:90')
plt.tight_layout()
save_plot(f'profile_sample{idx_end}_zoom.png')

# Sample end second profile (row 95, col 70:120)
plt.figure(figsize=(8, 4))
plt.plot(decoded[idx_end, 95, 70:120], label='Predicted')
plt.plot(X_test[idx_end, 95, 70:120],  label='Tip-convoluted')
plt.plot(Y_test[idx_end, 95, 70:120],  label='Ground-truth')
plt.legend()
plt.title(f'Line Profile — Sample {idx_end}, row 95, col 70:120')
plt.tight_layout()
save_plot(f'profile_sample{idx_end}_row95_zoom.png')


# ============================================================
# Save results
# ============================================================

# .npy format (Python compatible)
np.save(f'{OUTPUT_DIR}/Test_Dilated.npy', X_test)
np.save(f'{OUTPUT_DIR}/Test_Ground.npy', Y_test)
np.save(f'{OUTPUT_DIR}/Decoded.npy', decoded)

# .mat format (MATLAB compatible)
savemat(f'{OUTPUT_DIR}/Test_Dilated.mat', {"a": X_test})
savemat(f'{OUTPUT_DIR}/Test_Ground.mat',  {"a": Y_test})
savemat(f'{OUTPUT_DIR}/Decoded.mat',      {"a": decoded})

# ============================================================
# 儲存本次訓練摘要至 RUN_DIR
# ============================================================
import datetime

# metrics.txt
metrics_path = os.path.join(RUN_DIR, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(f"AFM Deep Learning — 訓練摘要\n")
    f.write(f"{'='*50}\n")
    f.write(f"訓練目錄  : {RUN_DIR}\n")
    f.write(f"完成時間  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"[訓練設定]\n")
    f.write(f"  Epochs      : {EPOCHS}\n")
    f.write(f"  Batch Size  : {BATCH_SIZE}\n")
    f.write(f"  Train 樣本  : {X_processed_train.shape[0]}  (梯形孔+圓柱孔)\n")
    f.write(f"  Val 樣本    : {X_processed_test.shape[0]}\n\n")
    f.write(f"[正規化設定]\n")
    f.write(f"  NORM_MIN    : {NORM_MIN} nm\n")
    f.write(f"  NORM_MAX    : {NORM_MAX} nm\n")
    f.write(f"  背景 0 nm   : → {(0-NORM_MIN)/(NORM_MAX-NORM_MIN):.3f}\n\n")
    f.write(f"[最終 Loss（正規化空間 [0,1]）]\n")
    f.write(f"  Train Loss  : {training_data.history['loss'][-1]:.6f}\n")
    f.write(f"  Val Loss    : {training_data.history['val_loss'][-1]:.6f}\n\n")
    f.write(f"[評估指標 (測試集，單位 nm，反正規化後)]\n")
    for k, v in metrics.items():
        f.write(f"  {k:<6}: {v:.6f}\n")
    f.write(f"\n[模型路徑]\n")
    f.write(f"  {MODEL_PATH}\n")
    f.write(f"  {WEIGHTS_PATH}\n")
print(f"  [摘要] {metrics_path}")

# config.txt  — 記錄探針與曲面設定
config_path = os.path.join(RUN_DIR, 'config.txt')
with open(config_path, 'w', encoding='utf-8') as f:
    f.write(f"[訓練探針設定]\n")
    if tip_source == 'synthetic_paraboloid':
        _edge_nm    = TIP_TRAIN_SIZE // 2 * SURFACE_SCALE_NM
        _edge_depth = -(_edge_nm ** 2) / (2.0 * TIP_RADIUS_NM)
        f.write(f"  類型             : 合成拋物線探針 (Paraboloid)\n")
        f.write(f"  TIP_RADIUS_NM    : {TIP_RADIUS_NM} nm\n")
        f.write(f"  TIP_TRAIN_SIZE   : {TIP_TRAIN_SIZE} px\n")
        f.write(f"  邊緣深度          : {_edge_depth:.1f} nm\n")
        f.write(f"  粒子高度上限      : < {abs(_edge_depth):.0f} nm（不飽和條件）\n\n")
    else:
        f.write(f"  類型             : 真實量測探針 (tip.mat)\n")
        f.write(f"  來源檔案         : {tip_source}\n")
        f.write(f"  訓練尺寸         : {tip_shape.shape[0]}×{tip_shape.shape[1]} px "
                f"@ {SURFACE_SCALE_NM} nm/px\n")
        f.write(f"  深度範圍         : [{tip_shape.min():.1f}, {tip_shape.max():.1f}] nm\n\n")
    f.write(f"[訓練策略]\n")
    f.write(f"  樣品類型         : 全凹洞（移除突起粒子，突起與孔洞物理行為相反）\n")
    f.write(f"  資料來源 1       : 梯形孔 (trapezoid_hole)  {N_TRAP} 張\n")
    f.write(f"  資料來源 2       : 圓柱孔 (cylinder_hole)   {N_CYL} 張（小+大混合尺度，圓形主形狀）\n")
    f.write(f"  星形孔           : 已移除（對圓形樣品增加角狀歧義，grey dilation 使形狀難區分）\n")
    f.write(f"  圓柱孔半徑       : 1.5–24 px（小孔 70% / 大孔 30%）\n")
    f.write(f"  深度範圍         : 60–145 nm\n")
    f.write(f"  SURFACE_SCALE_NM : {SURFACE_SCALE_NM} nm/unit\n")
    f.write(f"\n[梯形孔樣品設定（用戶真實樣品）]\n")
    f.write(f"  開口寬度  : {TRAP_OPEN_NM} nm  (±{int(TRAP_VARIATION*100)}%)\n")
    f.write(f"  底部寬度  : {TRAP_BOT_NM} nm  (±{int(TRAP_VARIATION*100)}%)\n")
    f.write(f"  深度      : {TRAP_DEPTH_NM} nm  (±{int(TRAP_VARIATION*100)}%)\n")
    f.write(f"  壁角      : {math.degrees(math.atan((TRAP_OPEN_NM-TRAP_BOT_NM)/2/TRAP_DEPTH_NM)):.1f}°\n")
    f.write(f"\n[全域正規化（訓練與推論必須一致）]\n")
    f.write(f"  NORM_MIN         : {NORM_MIN} nm\n")
    f.write(f"  NORM_MAX         : {NORM_MAX} nm\n")
    f.write(f"  x_norm = (x - NORM_MIN) / (NORM_MAX - NORM_MIN)\n")
    f.write(f"  背景 0 nm → {(0-NORM_MIN)/(NORM_MAX-NORM_MIN):.3f}（不是零，模型不能偷懶）\n")
    f.write(f"\n[估算探針（供 detect.py 推論使用）]\n")
    f.write(f"  tip_file         : tip.mat/tip_estimated0526.mat\n")
    f.write(f"  tip_correction   : tip_correction.py\n")
print(f"  [設定] {config_path}")

print(f"\n{'='*60}")
print(f"  訓練完成！所有結果已儲存至：{RUN_DIR}")
print(f"{'='*60}")
print("All results saved successfully.")