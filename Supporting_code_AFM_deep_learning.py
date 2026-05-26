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
from keras.layers import Activation, Input, Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
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
# TIP_TRAIN_SIZE = 9 px（at 39.1 nm/px）：
#   有效探針半徑 = 73 nm / 39.1 nm/px ≈ 1.87 px
#   偏移 3px：深度 = -(3×39.1)²/(2×73) = -94 nm（超過最深孔洞 →不影響）
#   故 9px（半徑 4px）已足夠完整捕捉 dilation 物理效應
# ---------------------------------------------------------------
TIP_RADIUS_NM  = 73.0   # 從 tip_estimated0526.mat 量得的真實探針 ROC (nm)
TIP_TRAIN_SIZE = 9      # 合成探針尺寸 (px)，奇數；9px × 39.1nm = ±156nm 物理範圍

# ---- 梯形孔樣品標稱參數（用於訓練資料生成）--------------------
# 來源：用戶實際掃描的校正樣品
TRAP_OPEN_NM  = 228.8   # 開口寬度 (nm)
TRAP_BOT_NM   = 183.5   # 底部寬度 (nm)
TRAP_DEPTH_NM = 125.8   # 深度 (nm)
TRAP_VARIATION = 0.20   # 隨機變異範圍 ±20%

# ---- 全域正規化常數（解決模型輸出全零的關鍵）---------------------
# 訓練資料物理範圍：
#   負值：梯形孔最深 = -125.8 × 1.2 ≈ -151 nm → 取 -155 nm（安全邊界）
#   正值：cubic/cylinder 最高 = 20 px × 5 nm = 100 nm → 取 105 nm
# 公式：x_norm = (x - NORM_MIN) / (NORM_MAX - NORM_MIN)  → [0, 1]
# 效果：背景 0 nm 正規化後 = 0.597（不再是特殊零值）
#       孔洞 -125 nm → 0.097，突起 100 nm → 0.984
#       模型無法靠「全零輸出」最小化 loss，被迫真正學習
# ---------------------------------------------------------------
NORM_MIN = -155.0   # nm（含安全邊界）
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
    n_holes = random.randint(1, 3)
    centers_px, radii_px, depths_nm = [], [], []

    for _ in range(n_holes):
        # ── 可見度門檻（at 39.1 nm/px）──────────────────────────────
        # r_min = sqrt(2 × R_tip × depth) / px_nm
        # R=73nm, depth≈100nm → r_min = sqrt(14600)/39.1 ≈ 3.1px
        # 取 3–8px 覆蓋臨界附近（117–313nm），增加難度多樣性
        r_px    = random.uniform(3, 8)           # 半徑 3–8 px → 117–313 nm
        depth   = random.uniform(80.0, 130.0)   # 深度 80–130 nm
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


# ---- 圓柱孔 N=400 ---------------------------------------------------
N = 400
list_cyl_hole_images = []

print(f"Generating cylinder holes (r=3–8px={3*SURFACE_SCALE_NM:.0f}–{8*SURFACE_SCALE_NM:.0f}nm, "
      f"depth=80–130nm, 1–3 holes/img)...")
for i in tqdm(range(N), desc="Cylinder Holes", unit="img",
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

N = 400

print(f"Generating trapezoidal holes "
      f"(open={TRAP_OPEN_NM}nm, bot={TRAP_BOT_NM}nm, "
      f"depth={TRAP_DEPTH_NM}nm, ±{int(TRAP_VARIATION*100)}%)...")
for i in tqdm(range(N), desc="Trapezoid", unit="img",
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


# ============================================================
# 套用 Grey Dilation（模擬 AFM 掃描，凹洞版本）
# Grey dilation 對孔洞的效果：
#   dilated[i,j] = max_{u,v}{ surface[i-u,j-v] + tip[u,v] }
#   孔洞（負值）+ tip（0 到負值）→ 孔洞在 dilated 中變淺/變窄
#   這正確模擬 AFM 探針無法完全進入孔洞的物理現象
# ============================================================

print("建立訓練用合成探針（拋物線，R=73nm）...")
tip_shape = make_training_tip()
plt.figure()
plt.imshow(tip_shape, cmap='viridis')
plt.colorbar()
plt.title(f"Training Tip — Paraboloid  R={TIP_RADIUS_NM} nm  "
          f"{TIP_TRAIN_SIZE}×{TIP_TRAIN_SIZE} px")
save_plot('tip_shape.png')

# 載入兩種孔洞 ground-truth
true_trap_surf_stack     = np.load(f'{OUTPUT_DIR}/true_trap_stack.npy').astype('float32')
true_cyl_hole_surf_stack = np.load(f'{OUTPUT_DIR}/true_cyl_hole_stack.npy').astype('float32')

list_trap_dilated_image     = []
list_cyl_hole_dilated_image = []

print("Applying dilation to hole images...")
for i in tqdm(range(400), desc="Dilation", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    true_trap_surf     = true_trap_surf_stack[i, :, :]
    true_cyl_hole_surf = true_cyl_hole_surf_stack[i, :, :]

    dilated_trap_surf     = scipy.ndimage.grey_dilation(true_trap_surf,     structure=tip_shape)
    dilated_cyl_hole_surf = scipy.ndimage.grey_dilation(true_cyl_hole_surf, structure=tip_shape)

    list_trap_dilated_image.append(dilated_trap_surf)
    list_cyl_hole_dilated_image.append(dilated_cyl_hole_surf)

dilated_trap_surf_stack     = np.stack(list_trap_dilated_image,     axis=0)
dilated_cyl_hole_surf_stack = np.stack(list_cyl_hole_dilated_image, axis=0)

np.save(f'{OUTPUT_DIR}/dilated_trap_stack.npy',     dilated_trap_surf_stack)
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
#   X1 / y1 : 梯形孔 (trapezoid_hole)  400 張
#   X2 / y2 : 圓柱孔 (cylinder_hole)   400 張
#   共 800 張，train:test = 8:2 → train 640, test 160 per type
#   合計 train 1280, test 320

dilated_images_path_trap     = f'{OUTPUT_DIR}/dilated_trap_stack.npy'
true_images_path_trap        = f'{OUTPUT_DIR}/true_trap_stack.npy'
dilated_images_path_cyl_hole = f'{OUTPUT_DIR}/dilated_cyl_hole_stack.npy'
true_images_path_cyl_hole    = f'{OUTPUT_DIR}/true_cyl_hole_stack.npy'

X1 = np.load(dilated_images_path_trap).astype('float32')
y1 = np.load(true_images_path_trap).astype('float32')
X2 = np.load(dilated_images_path_cyl_hole).astype('float32')
y2 = np.load(true_images_path_cyl_hole).astype('float32')

# Split
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_merged_train = np.concatenate((X_1_train, X_2_train), axis=0)
y_merged_train = np.concatenate((y_1_train, y_2_train), axis=0)

X_merged_train, y_merged_train = shuffle(X_merged_train, y_merged_train)

X_merged_test = np.concatenate((X_1_test, X_2_test), axis=0)
y_merged_test = np.concatenate((y_1_test, y_2_test), axis=0)

print("Training data loaded.",
      f"\n  ✓ Trapezoid holes : {X1.shape[0]} imgs",
      f"\n  ✓ Cylinder  holes : {X2.shape[0]} imgs",
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
# Neural Network Architecture (Convolutional Autoencoder)
# ============================================================

autoencoder = Sequential()
print(X_processed_train.shape[1:])

# Define the initializer
ini = tf.keras.initializers.HeNormal(1)

# Encoder Layers
autoencoder.add(Convolution2D(4,  (3, 3), kernel_initializer=ini, activation='relu',
                              padding='same', input_shape=X_processed_train.shape[1:]))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Convolution2D(8,  (3, 3), kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Convolution2D(16, (3, 3), kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Convolution2D(32, (3, 3), kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# Decoder Layers
autoencoder.add(Conv2DTranspose(32, (4, 4), strides=2, kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(16, (4, 4), strides=2, kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(8,  (4, 4), strides=2, kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(4,  (4, 4), strides=2, kernel_initializer=ini, activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(1,  (3, 3), kernel_initializer=ini,
                               activation='sigmoid', padding='same'))
# sigmoid 輸出恰好限制在 [0, 1]，與正規化資料範圍匹配
# 防止模型輸出超出範圍的數值

# List the training parameters and their information
autoencoder.summary()

# [Fix 7] 移除未使用的 input_img = Input(shape=(...))
# Apply the optimizer with desired learning rate and loss function
# learning_rate 從 0.001 → 0.0003：配合正規化後數值範圍較小，步進更穩定
opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
autoencoder.compile(optimizer=opt, loss='MAE')


# ============================================================
# Training
# ============================================================

EPOCHS = 2000
BATCH_SIZE = 32

progress_callback = TrainingProgressCallback(epochs=EPOCHS)

print(f"\n訓練參數:")
print(f"  Epochs:             {EPOCHS}")
print(f"  Batch Size:         {BATCH_SIZE}")
print(f"  Training Samples:   {X_processed_train.shape[0]}")
print(f"  Validation Samples: {X_processed_test.shape[0]}")
print(f"  Input Shape:        {X_processed_train.shape[1:]}\n")

training_data = autoencoder.fit(
    X_processed_train, y_processed_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_processed_test, y_processed_test),
    callbacks=[progress_callback],
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
_edge_nm     = TIP_TRAIN_SIZE // 2 * SURFACE_SCALE_NM
_edge_depth  = -(_edge_nm ** 2) / (2.0 * TIP_RADIUS_NM)
with open(config_path, 'w', encoding='utf-8') as f:
    f.write(f"[訓練探針設定]\n")
    f.write(f"  類型             : 合成拋物線探針 (Paraboloid)\n")
    f.write(f"  TIP_RADIUS_NM    : {TIP_RADIUS_NM} nm\n")
    f.write(f"  TIP_TRAIN_SIZE   : {TIP_TRAIN_SIZE} px\n")
    f.write(f"  邊緣深度          : {_edge_depth:.1f} nm\n")
    f.write(f"  粒子高度上限      : < {abs(_edge_depth):.0f} nm（不飽和條件）\n\n")
    f.write(f"[訓練策略]\n")
    f.write(f"  樣品類型         : 全凹洞（移除突起粒子，突起與孔洞物理行為相反）\n")
    f.write(f"  資料來源 1       : 梯形孔 (trapezoid_hole)  400 張\n")
    f.write(f"  資料來源 2       : 圓柱孔 (cylinder_hole)   400 張\n")
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