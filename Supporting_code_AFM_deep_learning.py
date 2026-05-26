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

# Create output directory
OUTPUT_DIR = 'img'
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# Simulate the spherical particles
# ============================================================
size = 128

# Create non-overlapping circles with randomized radii and centers
def circle_randomizer(max_attempts=10000):
    # [Fix 2] 加入最大嘗試次數，防止在圖像空間不足時陷入無限迴圈
    number_of_circles = random.randint(4, 10)
    radius_list = np.zeros(number_of_circles)
    center_coords = np.zeros(shape=(number_of_circles, 2))

    i = 0
    attempts = 0
    while i < number_of_circles:
        attempts += 1
        if attempts > max_attempts:
            # 若嘗試次數過多，縮減粒子數量後重試
            number_of_circles = i  # 保留已放置成功的粒子
            radius_list = radius_list[:number_of_circles]
            center_coords = center_coords[:number_of_circles]
            break

        radius_list[i] = random.uniform(8, 10)
        center_coords[i, 0] = random.uniform(0, size)
        center_coords[i, 1] = random.uniform(0, size)
        j = 0
        overlapping = False

        while j < i:
            dx = center_coords[i, 0] - center_coords[j, 0]
            dy = center_coords[i, 1] - center_coords[j, 1]
            dist2 = dx**2 + dy**2
            min_dist2 = (radius_list[i] + radius_list[j])**2
            if dist2 < min_dist2:
                overlapping = True
                break
            j += 1

        if overlapping:
            # 重新取樣此粒子，不改變 i
            continue

        i += 1

    return radius_list, center_coords


# [Fix 1] 重新命名為 spherical_image_creator，防止被後面的 cubic 版本覆蓋
# [Fix 4] arange 從 0 開始、迴圈涵蓋 0..size-1，修正邊緣像素缺漏
def spherical_image_creator(centers, radii, size):
    x = np.arange(0, size, 1)  # 0..size-1
    y = np.arange(0, size, 1)

    img_data = np.zeros(shape=(size, size))

    for c in range(len(centers)):
        x0 = centers[c, 0]
        y0 = centers[c, 1]
        radius = radii[c]
        for i in range(0, size):      # [Fix 4] 涵蓋全部行
            for j in range(0, size):  # [Fix 4] 涵蓋全部列
                upheight2 = radius**2 - (x[i] - x0)**2 - (y[j] - y0)**2
                if upheight2 >= 0:
                    img_data[i, j] = radius + sqrt(upheight2)

    return img_data

radius_list, center_coords = circle_randomizer()
img_data = spherical_image_creator(center_coords, radius_list, size)

N = 400

list_of_true_surf = []
list_of_dilated_surf = []

# Use the functions above to generate ground-truth and tip-convoluted pairs
print("Generating spherical particles...")
for i in tqdm(range(N), desc="Spherical", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    radii, centers = circle_randomizer()
    true_surface = spherical_image_creator(centers, radii, size)  # [Fix 1]
    true_surface = true_surface.astype(int)
    list_of_true_surf.append(true_surface)

# Save the images
true_spherical_surf_stack = np.stack(list_of_true_surf, axis=0)
np.save(f'{OUTPUT_DIR}/true_spherical_stack.npy', true_spherical_surf_stack)

# Plot the images
# [Fix 12] 補上 tight_layout 與 show
figure, axis = plt.subplots(1, 2, sharey=True)
axis[0].imshow(true_spherical_surf_stack[0])
axis[0].set_title("ground-truth (spherical)")
axis[1].set_title("(dilated will appear after dilation step)")
axis[1].axis('off')
plt.tight_layout()
plt.show()


# ============================================================
# Simulate the cuboidal particles
# ============================================================

def cubic_randomizer(max_attempts=10000):
    # [Fix 2] 加入最大嘗試次數，防止無限迴圈
    number_of_rects = random.randint(4, 10)
    l_list = np.zeros(number_of_rects)
    w_list = np.zeros(number_of_rects)
    h_list = np.zeros(number_of_rects)
    center_coords = np.zeros(shape=(number_of_rects, 2))

    i = 0
    attempts = 0
    while i < number_of_rects:
        attempts += 1
        if attempts > max_attempts:
            number_of_rects = i
            l_list = l_list[:number_of_rects]
            w_list = w_list[:number_of_rects]
            h_list = h_list[:number_of_rects]
            center_coords = center_coords[:number_of_rects]
            break

        l_list[i] = random.uniform(8, 10)
        w_list[i] = random.uniform(8, 10)
        h_list[i] = random.uniform(16, 20)
        center_coords[i, 0] = random.uniform(0, size)
        center_coords[i, 1] = random.uniform(0, size)

        overlapping = False
        for j in range(i):
            dx = abs(center_coords[i, 0] - center_coords[j, 0])  # [Fix 3] 使用 abs()
            dy = abs(center_coords[i, 1] - center_coords[j, 1])  # [Fix 3]
            if dx <= (l_list[i] + l_list[j]) and dy <= (w_list[i] + w_list[j]):
                overlapping = True
                break

        if overlapping:
            continue

        i += 1

    return l_list, w_list, h_list, center_coords


# [Fix 4] cubic image_creator：arange 從 0 開始、迴圈涵蓋全部像素
def cubic_image_creator(l, w, h, centers, size):
    x = np.arange(0, size, 1)  # [Fix 4]
    y = np.arange(0, size, 1)
    img_data = np.zeros(shape=(size, size))
    for c in range(len(centers)):
        x0 = centers[c, 0]
        y0 = centers[c, 1]
        l0 = l[c]
        w0 = w[c]
        h0 = h[c]
        for i in range(0, size):      # [Fix 4]
            for j in range(0, size):
                if abs(x[i] - x0) <= l0 and abs(y[j] - y0) <= w0:
                    img_data[i, j] = h0
    return img_data


l_list, w_list, h_list, center_coords = cubic_randomizer()
img_data = cubic_image_creator(l_list, w_list, h_list, center_coords, size)

N = 400

list_of_true_surf = []
list_of_dilated_surf = []

print("Generating cubic particles...")
for i in tqdm(range(N), desc="Cubic", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    l_list, w_list, h_list, center_coords = cubic_randomizer()
    true_surface = cubic_image_creator(l_list, w_list, h_list, center_coords, size)
    true_surface = true_surface.astype(int)
    list_of_true_surf.append(true_surface)

true_cubic_surf_stack = np.stack(list_of_true_surf, axis=0)
np.save(f'{OUTPUT_DIR}/true_cubic_stack.npy', true_cubic_surf_stack)

# [Fix 12]
figure, axis = plt.subplots(1, 2, sharey=True)
axis[0].imshow(true_cubic_surf_stack[0])
axis[0].set_title("ground-truth (cubic)")
axis[1].set_title("(dilated will appear after dilation step)")
axis[1].axis('off')
plt.tight_layout()
plt.show()


# ============================================================
# Simulate the cylindrical particles
# ============================================================

list_poly_gndtruth_image = []
list_poly_dilated_image = []

N = 400

# Background subtraction to ensure height consistency
background = np.full((128, 128), -255)

print("Generating cylindrical particles...")
for i in tqdm(range(N), desc="Cylindrical", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    result = random_shapes((128, 128), max_shapes=10, min_shapes=4,
                           min_size=16, max_size=20, intensity_range=((235, 239),))
    surface, labels = result
    poly_gndtruth_image = np.negative(np.add(surface[:, :, 0], background)).astype(int)
    list_poly_gndtruth_image.append(poly_gndtruth_image)

true_poly_surf_stack = np.stack(list_poly_gndtruth_image, axis=0)
np.save(f'{OUTPUT_DIR}/true_poly_stack.npy', true_poly_surf_stack)

# [Fix 12]
figure, axis = plt.subplots(1, 2, sharey=True)
axis[0].imshow(true_poly_surf_stack[0])
axis[0].set_title("ground-truth (cylindrical)")
axis[1].set_title("(dilated will appear after dilation step)")
axis[1].axis('off')
plt.tight_layout()
plt.show()


# ============================================================
# Import AFM tip and apply grey dilation
# ============================================================

tip = io.loadmat('tip.mat/tip_estimated0526.mat')
tip_shape = tip['tip']
plt.imshow(tip_shape)
plt.colorbar()
plt.title("AFM Tip Shape")
plt.show()

# Load saved stacks
true_spherical_surf_stack = np.load(f'{OUTPUT_DIR}/true_spherical_stack.npy').astype('float32')
true_cubic_surf_stack     = np.load(f'{OUTPUT_DIR}/true_cubic_stack.npy').astype('float32')
true_poly_surf_stack      = np.load(f'{OUTPUT_DIR}/true_poly_stack.npy').astype('float32')

list_spherical_dilated_image = []
list_cubic_dilated_image = []
list_poly_dilated_image = []

print("Applying dilation to images...")
for i in tqdm(range(400), desc="Dilation", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    true_spherical_surf = true_spherical_surf_stack[i, :, :]
    true_cubic_surf     = true_cubic_surf_stack[i, :, :]
    true_poly_surf      = true_poly_surf_stack[i, :, :]

    dilated_spherical_surf = scipy.ndimage.grey_dilation(true_spherical_surf, structure=tip_shape)
    dilated_cubic_surf     = scipy.ndimage.grey_dilation(true_cubic_surf, structure=tip_shape)
    dilated_poly_surf      = scipy.ndimage.grey_dilation(true_poly_surf, structure=tip_shape)

    list_spherical_dilated_image.append(dilated_spherical_surf)
    list_cubic_dilated_image.append(dilated_cubic_surf)
    list_poly_dilated_image.append(dilated_poly_surf)

dilated_spherical_surf_stack = np.stack(list_spherical_dilated_image, axis=0)
dilated_cubic_surf_stack     = np.stack(list_cubic_dilated_image, axis=0)
dilated_poly_surf_stack      = np.stack(list_poly_dilated_image, axis=0)

np.save(f'{OUTPUT_DIR}/dilated_spherical_stack.npy', dilated_spherical_surf_stack)
np.save(f'{OUTPUT_DIR}/dilated_cubic_stack.npy', dilated_cubic_surf_stack)
np.save(f'{OUTPUT_DIR}/dilated_poly_stack.npy', dilated_poly_surf_stack)


# ============================================================
# Add line-like artifacts (post-dilation)
# ============================================================

def line_artifact_adder(image_data, image_size):
    # [Fix 5] 修正方向：AFM 掃描偽影為水平（row-based），
    #         line_picker shape 從 (128,1) 改為 (1,128)
    # 背景值 0.0：新 tip max=0nm，dilation 不產生偏移，無需校正
    processed_images_l = []
    for image in tqdm(image_data, desc="Line Artifacts", unit="img", leave=False):
        processed_image = image + np.full((128, 128), 0.0)  # Background subtraction
        line_picker = np.random.random((1, 128))                 # [Fix 5] (1, 128) → row mask
        line_picker = line_picker > 0.05
        preserved_lines = np.ones((128, 128)) * line_picker      # 廣播：每一行整排同值
        processed_image = processed_image * preserved_lines
        processed_images_l.append(processed_image.reshape(image_size[0], image_size[1], 1))
    processed_images_a = np.stack(processed_images_l, axis=0)
    return processed_images_a

dilated_spherical_surf_stack       = np.load(f'{OUTPUT_DIR}/dilated_spherical_stack.npy').astype('float32')
dilated_edge_lifted_cubic_surf_stack = np.load(f'{OUTPUT_DIR}/dilated_cubic_stack.npy').astype('float32')
dilated_edge_lifted_poly_surf_stack  = np.load(f'{OUTPUT_DIR}/dilated_poly_stack.npy').astype('float32')

target_size = (128, 128)
spherical_dilated_line_artifact = line_artifact_adder(dilated_spherical_surf_stack, target_size)
cubic_dilated_line_artifact     = line_artifact_adder(dilated_edge_lifted_cubic_surf_stack, target_size)
poly_dilated_line_artifact      = line_artifact_adder(dilated_edge_lifted_poly_surf_stack, target_size)

np.save(f'{OUTPUT_DIR}/dilated_spherical_incl_line_artifact.npy', spherical_dilated_line_artifact)
np.save(f'{OUTPUT_DIR}/dilated_cubic_incl_line_artifact.npy', cubic_dilated_line_artifact)
np.save(f'{OUTPUT_DIR}/dilated_poly_incl_line_artifact.npy', poly_dilated_line_artifact)


# ============================================================
# Add edge-lift artifacts (prior to dilation)
# ============================================================

def edge_lifter(image_data, image_size):
    import cv2
    processed_images_l = []
    for image in tqdm(image_data, desc="Edge Lifting", unit="img", leave=False):
        processed_image = image

        # Detect the edges
        img = image.astype(np.uint8)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian2 = laplacian < 0

        # Raise the edges by 1 nm
        processed_image = image + laplacian2 * 1
        processed_images_l.append(processed_image.reshape(image_size[0], image_size[1]))
    processed_images_a = np.stack(processed_images_l, axis=0)
    return processed_images_a

target_size = (128, 128)
print("Loading cubic and poly stacks for edge lifting...")
true_cubic_surf_stack = np.load(f'{OUTPUT_DIR}/true_cubic_stack.npy').astype('float32')
true_poly_surf_stack  = np.load(f'{OUTPUT_DIR}/true_poly_stack.npy').astype('float32')
edge_lifted_cubic_stack = edge_lifter(true_cubic_surf_stack, target_size)
edge_lifted_poly_stack  = edge_lifter(true_poly_surf_stack, target_size)
np.save(f'{OUTPUT_DIR}/edge_lifted_cubic_stack.npy', edge_lifted_cubic_stack)
np.save(f'{OUTPUT_DIR}/edge_lifted_poly_stack.npy', edge_lifted_poly_stack)

tip = io.loadmat('tip.mat/tip_estimated.mat')
tip_shape = tip['tip']
plt.imshow(tip_shape)
plt.colorbar()
plt.title("AFM Tip Shape")
plt.show()

# Dilating the edge-lifted cubic and poly stacks
edge_lifted_cubic_surf_stack = np.load(f'{OUTPUT_DIR}/edge_lifted_cubic_stack.npy').astype('float32')
edge_lifted_poly_surf_stack  = np.load(f'{OUTPUT_DIR}/edge_lifted_poly_stack.npy').astype('float32')

list_cubic_dilated_image = []
list_poly_dilated_image = []

print("Dilating edge-lifted images...")
for i in tqdm(range(400), desc="Edge Dilation", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    edge_lifted_cubic_surf = edge_lifted_cubic_surf_stack[i, :, :]
    edge_lifted_poly_surf  = edge_lifted_poly_surf_stack[i, :, :]

    dilated_cubic_surf = scipy.ndimage.grey_dilation(edge_lifted_cubic_surf, structure=tip_shape)
    dilated_poly_surf  = scipy.ndimage.grey_dilation(edge_lifted_poly_surf, structure=tip_shape)

    list_cubic_dilated_image.append(dilated_cubic_surf)
    list_poly_dilated_image.append(dilated_poly_surf)

dilated_cubic_surf_stack = np.stack(list_cubic_dilated_image, axis=0)
dilated_poly_surf_stack  = np.stack(list_poly_dilated_image, axis=0)

np.save(f'{OUTPUT_DIR}/dilated_edge_lifted_cubic_stack.npy', dilated_cubic_surf_stack)
np.save(f'{OUTPUT_DIR}/dilated_edge_lifted_poly_stack.npy', dilated_poly_surf_stack)


# ============================================================
# Load images and prepare training/testing datasets
# ============================================================

# [Fix 10] 修正路徑變數命名：
#   dilated_images_path_* = 卷積圖 (模型輸入 X)
#   true_images_path_*    = 真實表面 (模型標籤 y)

dilated_images_path_spherical = f'{OUTPUT_DIR}/dilated_spherical_stack.npy'
true_images_path_spherical    = f'{OUTPUT_DIR}/true_spherical_stack.npy'
dilated_images_path_cubic     = f'{OUTPUT_DIR}/dilated_cubic_stack.npy'
true_images_path_cubic        = f'{OUTPUT_DIR}/true_cubic_stack.npy'
dilated_images_path_poly      = f'{OUTPUT_DIR}/dilated_poly_stack.npy'
true_images_path_poly         = f'{OUTPUT_DIR}/true_poly_stack.npy'

X1 = np.load(dilated_images_path_spherical).astype('float32')  # Input:  tip-convoluted
y1 = np.load(true_images_path_spherical).astype('float32')     # Label:  true surface
X2 = np.load(dilated_images_path_cubic).astype('float32')
y2 = np.load(true_images_path_cubic).astype('float32')
X3 = np.load(dilated_images_path_poly).astype('float32')
y3 = np.load(true_images_path_poly).astype('float32')

# Split the entire dataset to training and testing dataset
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

X_merged_train = np.concatenate((X_1_train, X_2_train, X_3_train), axis=0)
y_merged_train = np.concatenate((y_1_train, y_2_train, y_3_train), axis=0)

# Shuffle the training dataset only to help improve the model's performance
X_merged_train, y_merged_train = shuffle(X_merged_train, y_merged_train)

X_merged_test = np.concatenate((X_1_test, X_2_test, X_3_test), axis=0)
y_merged_test = np.concatenate((y_1_test, y_2_test, y_3_test), axis=0)

print("Training data loaded.",
      "\nno. of images in the training set:", str(X_merged_train.shape[0]),
      "\nno. of images in the testing set:", str(X_merged_test.shape[0]),
      "\nresolution of each image:", str((X_merged_train.shape[1], X_merged_train.shape[2])))


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
plt.show()


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
autoencoder.add(Conv2DTranspose(1,  (3, 3), kernel_initializer=ini, padding='same'))

# List the training parameters and their information
autoencoder.summary()

# [Fix 7] 移除未使用的 input_img = Input(shape=(...))
# Apply the optimizer with desired learning rate and loss function
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
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

MODEL_PATH = f'{OUTPUT_DIR}/AFM_MAE_autoencoder.keras'
WEIGHTS_PATH = f'{OUTPUT_DIR}/AFM_MAE_autoencoder.weights.h5'

autoencoder.save(MODEL_PATH)
autoencoder.save_weights(WEIGHTS_PATH)

np.save(f'{OUTPUT_DIR}/train_history.npy', training_data.history)
history = np.load(f'{OUTPUT_DIR}/train_history.npy', allow_pickle=True).item()

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
plt.show()


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
    # Peak Signal-to-Noise Ratio
    max_pixel = np.max(y_true)
    psnr = 20 * np.log10(max_pixel / rmse) if rmse != 0 else float('inf')
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

# 計算並顯示預測評估指標
print("\n" + "="*70)
print("預測結果評估")
print("="*70)
metrics = calculate_metrics(y_processed_test, decoded_imgs)
print(f"\n測試集評估指標:")
print(f"  MAE  (平均絕對誤差): {metrics['MAE']:.6f} nm")
print(f"  MSE  (均方誤差):      {metrics['MSE']:.6f} nm²")
print(f"  RMSE (均方根誤差):    {metrics['RMSE']:.6f} nm")
print(f"  PSNR (峰值信噪比):    {metrics['PSNR']:.2f} dB")
print(f"  SSIM (結構相似性):    {metrics['SSIM']:.4f}  [0=差, 1=完美]")

# 計算像素級準確率 (使用閾值)
threshold = 0.1  # 10% 誤差容限
diff = np.abs(y_processed_test - decoded_imgs)
max_val = np.max(y_processed_test)
accuracy = np.mean((diff / (max_val + 1e-10)) < threshold) * 100
print(f"  Pixel Accuracy (<10% error): {accuracy:.2f}%")
print("="*70 + "\n")


# ============================================================
# Visualize predictions (2×3 grid)
# ============================================================

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

for col in range(3):
    # Row 0: model output
    img_out = decoded_imgs[col].reshape(target_size[0], target_size[1])
    ax[0, col].imshow(img_out, cmap='gray')
    ax[0, col].set_title(f'Output {col + 1}')
    ax[0, col].axis('off')
    # Row 1: model input
    img_in = X_processed_test[col].reshape(target_size[0], target_size[1])
    ax[1, col].imshow(img_in, cmap='gray')
    ax[1, col].set_title(f'Input {col + 1}')
    ax[1, col].axis('off')

plt.tight_layout()
plt.show(block=True)


# ============================================================
# Load saved model and make further predictions for visualization
# [Fix 6] 載入路徑改為 .keras
# ============================================================

model = load_model(MODEL_PATH)
decoded = model.predict(X_processed_test)

# Squeeze channel dimension for visualization
X_test  = X_processed_test.squeeze()
Y_test  = y_processed_test.squeeze()
decoded = decoded.squeeze()


def show_triplet(idx, row_slice=None, col_slice=slice(None), label=""):
    """
    顯示單一樣本的 tip-convoluted / ground-truth / predicted 三張圖，
    並繪製指定切線的剖面圖。
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [X_test[idx], Y_test[idx], decoded[idx]],
                                ['Tip-convoluted', 'Ground-truth', 'Predicted']):
        im = ax.imshow(data)
        ax.set_title(f'{title} [{label}]')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

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
        plt.show()


# Sample 0 (row 65, full width)
show_triplet(0, row_slice=65, label="sample 0")

# Sample 100 (row 60, full width + zoomed 25:100)
show_triplet(100, row_slice=60, label="sample 100")
# Zoomed view
plt.figure(figsize=(8, 4))
plt.plot(decoded[100, 60, 25:100], label='Predicted')
plt.plot(X_test[100, 60, 25:100],  label='Tip-convoluted')
plt.plot(Y_test[100, 60, 25:100],  label='Ground-truth')
plt.legend()
plt.title('Line Profile — Sample 100, row 60, col 25:100')
plt.tight_layout()
plt.show()

# Sample 200 (row 20, full width + zoomed 30:90)
show_triplet(200, row_slice=20, label="sample 200")
plt.figure(figsize=(8, 4))
plt.plot(decoded[200, 20, 30:90], label='Predicted')
plt.plot(X_test[200, 20, 30:90],  label='Tip-convoluted')
plt.plot(Y_test[200, 20, 30:90],  label='Ground-truth')
plt.legend()
plt.title('Line Profile — Sample 200, row 20, col 30:90')
plt.tight_layout()
plt.show()

# Sample 200 second profile (row 95, col 70:120)
plt.figure(figsize=(8, 4))
plt.plot(decoded[200, 95, 70:120], label='Predicted')
plt.plot(X_test[200, 95, 70:120],  label='Tip-convoluted')
plt.plot(Y_test[200, 95, 70:120],  label='Ground-truth')
plt.legend()
plt.title('Line Profile — Sample 200, row 95, col 70:120')
plt.tight_layout()
plt.show()


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

print("All results saved successfully.")