#!/usr/bin/env python3
"""
AFM 模型評估程式
載入訓練好的模型權重，對測試資料集進行預測，輸出完整評估指標。

用法:
  python evaluate.py                         # 使用預設路徑
  python evaluate.py --model img/AFM_MAE_autoencoder.keras
  python evaluate.py --weights img/AFM_MAE_autoencoder.weights.h5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import (Convolution2D, MaxPooling2D,
                                     Conv2DTranspose)
from skimage.metrics import structural_similarity as ssim_func
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.io import savemat

OUTPUT_DIR = 'img'
TARGET_SIZE = (128, 128)


# ── GPU 設定 ─────────────────────────────────────────────────────────────────

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU 已啟用: {gpus}")
    else:
        print("⚠ 使用 CPU 進行推理")


# ── 模型架構（與訓練時完全相同）──────────────────────────────────────────────

def build_model(input_shape):
    ini = tf.keras.initializers.HeNormal(1)
    model = Sequential([
        Convolution2D(4,  (3,3), kernel_initializer=ini, activation='relu',
                      padding='same', input_shape=input_shape),
        MaxPooling2D((2,2), padding='same'),
        Convolution2D(8,  (3,3), kernel_initializer=ini, activation='relu', padding='same'),
        MaxPooling2D((2,2), padding='same'),
        Convolution2D(16, (3,3), kernel_initializer=ini, activation='relu', padding='same'),
        MaxPooling2D((2,2), padding='same'),
        Convolution2D(32, (3,3), kernel_initializer=ini, activation='relu', padding='same'),
        MaxPooling2D((2,2), padding='same'),
        Conv2DTranspose(32, (4,4), strides=2, kernel_initializer=ini, activation='relu', padding='same'),
        Conv2DTranspose(16, (4,4), strides=2, kernel_initializer=ini, activation='relu', padding='same'),
        Conv2DTranspose(8,  (4,4), strides=2, kernel_initializer=ini, activation='relu', padding='same'),
        Conv2DTranspose(4,  (4,4), strides=2, kernel_initializer=ini, activation='relu', padding='same'),
        Conv2DTranspose(1,  (3,3), kernel_initializer=ini, padding='same'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='MAE')
    return model


# ── 資料載入（優先使用已存檔的測試集，否則從原始 stacks 重建）────────────────

def load_test_data():
    """
    回傳 (X_test_4d, y_test_4d)  shape: (N, 128, 128, 1)
    優先使用 Test_Dilated.npy / Test_Ground.npy（訓練後已儲存的 3D）。
    若不存在，則從 dilated/true stacks 重建（需要原始資料）。
    """
    x_path = os.path.join(OUTPUT_DIR, 'Test_Dilated.npy')
    y_path = os.path.join(OUTPUT_DIR, 'Test_Ground.npy')

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"  載入已存檔測試集: {x_path}")
        X = np.load(x_path).astype('float32')   # (N,128,128)
        Y = np.load(y_path).astype('float32')
        # 補回 channel 維度
        X = X[..., np.newaxis]
        Y = Y[..., np.newaxis]
        return X, Y

    # 從原始 stacks 重建
    print("  找不到 Test_Dilated.npy，嘗試從原始 stacks 重建測試集...")
    required = [
        'dilated_spherical_stack.npy', 'true_spherical_stack.npy',
        'dilated_cubic_stack.npy',     'true_cubic_stack.npy',
        'dilated_poly_stack.npy',      'true_poly_stack.npy',
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if missing:
        print(f"  ❌ 缺少檔案: {missing}")
        print(f"     請先執行 Supporting_code_AFM_deep_learning.py 產生資料與模型。")
        sys.exit(1)

    def load(name):
        return np.load(os.path.join(OUTPUT_DIR, name)).astype('float32')

    X1, y1 = load('dilated_spherical_stack.npy'), load('true_spherical_stack.npy')
    X2, y2 = load('dilated_cubic_stack.npy'),     load('true_cubic_stack.npy')
    X3, y3 = load('dilated_poly_stack.npy'),       load('true_poly_stack.npy')

    def split_test(X, y):
        _, Xt, _, yt = train_test_split(X, y, test_size=0.2, random_state=42)
        return Xt, yt

    Xt1, yt1 = split_test(X1, y1)
    Xt2, yt2 = split_test(X2, y2)
    Xt3, yt3 = split_test(X3, y3)

    X_test_raw = np.concatenate([Xt1, Xt2, Xt3], axis=0)
    y_test_raw = np.concatenate([yt1, yt2, yt3], axis=0)

    # 同訓練時的前處理
    def preprocess_X(data):
        return (data + np.full(TARGET_SIZE, -167.73)).reshape(
            data.shape[0], TARGET_SIZE[0], TARGET_SIZE[1], 1)

    def preprocess_y(data):
        return data.reshape(data.shape[0], TARGET_SIZE[0], TARGET_SIZE[1], 1)

    return preprocess_X(X_test_raw), preprocess_y(y_test_raw)


# ── 評估指標 ─────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    max_pixel = float(np.max(y_true))
    psnr = (20.0 * np.log10(max_pixel / rmse)) if rmse > 0 else float('inf')

    sq_t = y_true.squeeze()   # (N,128,128)
    sq_p = y_pred.squeeze()
    data_range = float(sq_t.max() - sq_t.min())
    ssim_vals = [ssim_func(sq_t[k], sq_p[k], data_range=data_range)
                 for k in range(len(sq_t))]
    ssim = float(np.mean(ssim_vals))

    threshold = 0.1
    diff = np.abs(y_true - y_pred)
    pixel_acc = float(np.mean((diff / (max_pixel + 1e-10)) < threshold) * 100)

    return {
        'MAE':        mae,
        'MSE':        mse,
        'RMSE':       rmse,
        'PSNR':       psnr,
        'SSIM':       ssim,
        'PixelAcc':   pixel_acc,
        'N_samples':  len(y_true),
    }


def print_metrics(metrics):
    n = metrics['N_samples']
    print("\n" + "="*70)
    print(f"  預測評估結果   (測試樣本數: {n})")
    print("="*70)
    print(f"  MAE  (平均絕對誤差):      {metrics['MAE']:.6f}  nm")
    print(f"  MSE  (均方誤差):          {metrics['MSE']:.6f}  nm²")
    print(f"  RMSE (均方根誤差):        {metrics['RMSE']:.6f}  nm")
    print(f"  PSNR (峰值信噪比):        {metrics['PSNR']:.4f}  dB")
    print(f"  SSIM (結構相似性):        {metrics['SSIM']:.4f}  [0=差, 1=完美]")
    print(f"  Pixel Accuracy (<10%):    {metrics['PixelAcc']:.2f}  %")
    print("="*70)


# ── 視覺化 ────────────────────────────────────────────────────────────────────

def plot_loss_curve():
    hist_path = os.path.join(OUTPUT_DIR, 'train_history.npy')
    if not os.path.exists(hist_path):
        print("  (找不到 train_history.npy，跳過 Loss 曲線)")
        return
    history = np.load(hist_path, allow_pickle=True).item()
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss'],     label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'), dpi=150)
    plt.show()
    print(f"  Loss 曲線已儲存: {OUTPUT_DIR}/loss_curve.png")


def plot_predictions(X_test, y_test, decoded, n_samples=3):
    X_sq = X_test.squeeze()
    Y_sq = y_test.squeeze()
    D_sq = decoded.squeeze()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    titles = ['Input (Dilated)', 'Ground Truth', 'Predicted']

    for row in range(n_samples):
        idx = row * (len(X_sq) // n_samples)
        for col, (data, title) in enumerate(zip(
                [X_sq[idx], Y_sq[idx], D_sq[idx]], titles)):
            im = axes[row, col].imshow(data, cmap='viridis')
            axes[row, col].set_title(f'{title} [#{idx}]')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.suptitle('Prediction Results', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'prediction_samples.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  預測範例圖已儲存: {save_path}")


def plot_error_histogram(y_test, decoded):
    diff = (y_test - decoded).flatten()
    plt.figure(figsize=(7, 4))
    plt.hist(diff, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', label='Zero error')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (nm)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'error_histogram.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  誤差分布圖已儲存: {save_path}")


# ── 儲存結果 ──────────────────────────────────────────────────────────────────

def save_metrics_csv(metrics):
    import csv
    path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Unit'])
        writer.writerow(['MAE',        f"{metrics['MAE']:.6f}",      'nm'])
        writer.writerow(['MSE',        f"{metrics['MSE']:.6f}",      'nm²'])
        writer.writerow(['RMSE',       f"{metrics['RMSE']:.6f}",     'nm'])
        writer.writerow(['PSNR',       f"{metrics['PSNR']:.4f}",     'dB'])
        writer.writerow(['SSIM',       f"{metrics['SSIM']:.4f}",     ''])
        writer.writerow(['PixelAcc',   f"{metrics['PixelAcc']:.2f}", '%'])
        writer.writerow(['N_samples',  metrics['N_samples'],          ''])
    print(f"  評估指標已儲存: {path}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AFM 模型評估')
    parser.add_argument('--model',   default=os.path.join(OUTPUT_DIR, 'AFM_MAE_autoencoder.keras'),
                        help='完整模型路徑 (.keras)')
    parser.add_argument('--weights', default=None,
                        help='僅載入權重 (.h5)，需搭配重建模型架構')
    parser.add_argument('--no-plot', action='store_true', help='不顯示圖表')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  AFM 深度學習模型 — 評估程式")
    print("="*70)

    setup_gpu()

    # 1. 載入測試資料
    print("\n[1] 載入測試資料...")
    X_test, y_test = load_test_data()
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # 2. 載入模型
    print("\n[2] 載入模型...")
    if args.weights:
        # 僅載入權重：重建架構後載入
        if not os.path.exists(args.weights):
            print(f"  ❌ 找不到權重檔案: {args.weights}")
            sys.exit(1)
        print(f"  從權重檔建立模型: {args.weights}")
        model = build_model(input_shape=X_test.shape[1:])
        model.load_weights(args.weights)
        print("  ✓ 權重載入完成")
    else:
        if not os.path.exists(args.model):
            print(f"  ❌ 找不到模型檔案: {args.model}")
            print(f"     請先執行 Supporting_code_AFM_deep_learning.py 完成訓練。")
            sys.exit(1)
        print(f"  載入完整模型: {args.model}")
        model = load_model(args.model)
        print("  ✓ 模型載入完成")

    model.summary()

    # 3. 預測
    print(f"\n[3] 對 {len(X_test)} 張測試影像進行預測...")
    decoded = model.predict(X_test, batch_size=32, verbose=1)
    print(f"  預測輸出 shape: {decoded.shape}")
    print(f"  輸出數值範圍: [{decoded.min():.4f}, {decoded.max():.4f}] nm")

    # 4. 計算評估指標
    print("\n[4] 計算評估指標...")
    metrics = calculate_metrics(y_test, decoded)
    print_metrics(metrics)

    # 5. 儲存指標 CSV
    save_metrics_csv(metrics)

    # 6. 圖表
    if not args.no_plot:
        print("\n[5] 繪製結果圖表...")
        plot_loss_curve()
        plot_predictions(X_test, y_test, decoded, n_samples=3)
        plot_error_histogram(y_test, decoded)

    print("\n✓ 評估完成！")


if __name__ == '__main__':
    main()
