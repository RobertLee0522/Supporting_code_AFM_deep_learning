#!/usr/bin/env python3
"""
AFM 影像預測程式
使用訓練好的 Autoencoder 模型對新的 AFM 影像進行去卷積預測
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.io import savemat

# 設定參數
OUTPUT_DIR = 'img'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'AFM_MAE_autoencoder.keras')
TARGET_SIZE = (128, 128)


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
    預處理輸入影像

    Args:
        image_path: 輸入影像路徑 (.tif, .png, .jpg 等)

    Returns:
        processed_image: 預處理後的影像 (1, 128, 128, 1)
        original_size: 原始影像尺寸
    """
    # 讀取影像
    img = Image.open(image_path)
    original_size = img.size

    # 轉換為灰階 (如果是彩色影像)
    if img.mode != 'L':
        img = img.convert('L')

    # Resize 到 128x128
    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # 轉換為 numpy array
    img_array = np.array(img_resized, dtype=np.float32)

    # 背景減去 (與訓練時相同的處理)
    # 注意：這裡假設輸入影像的數值範圍與訓練資料相近
    processed_image = img_array + np.full(TARGET_SIZE, -167.73)

    # Reshape 為模型期望的格式 (batch_size, height, width, channels)
    processed_image = processed_image.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 1)

    return processed_image, original_size, np.array(img_resized)


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
    predicted = model.predict(input_image, verbose=1)
    return predicted


def save_results(input_image, predicted_image, output_prefix):
    """
    儲存預測結果

    Args:
        input_image: 原始輸入影像
        predicted_image: 預測結果
        output_prefix: 輸出檔案前綴
    """
    output_dir = os.path.join(OUTPUT_DIR, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    # 提取資料 (移除 batch 和 channel 維度)
    input_2d = input_image[0, :, :, 0]
    predicted_2d = predicted_image[0, :, :, 0]

    # 儲存為 .npy
    np.save(os.path.join(output_dir, f'{output_prefix}_input.npy'), input_2d)
    np.save(os.path.join(output_dir, f'{output_prefix}_predicted.npy'), predicted_2d)

    # 儲存為 .mat (MATLAB 格式)
    savemat(os.path.join(output_dir, f'{output_prefix}_input.mat'), {'image': input_2d})
    savemat(os.path.join(output_dir, f'{output_prefix}_predicted.mat'), {'image': predicted_2d})

    # 儲存為影像 (.png)
    # 正規化到 0-255 範圍
    input_norm = ((input_2d - input_2d.min()) / (input_2d.max() - input_2d.min()) * 255).astype(np.uint8)
    predicted_norm = ((predicted_2d - predicted_2d.min()) / (predicted_2d.max() - predicted_2d.min()) * 255).astype(np.uint8)

    Image.fromarray(input_norm).save(os.path.join(output_dir, f'{output_prefix}_input.png'))
    Image.fromarray(predicted_norm).save(os.path.join(output_dir, f'{output_prefix}_predicted.png'))

    print(f"\n結果已儲存至: {output_dir}/")
    print(f"  - {output_prefix}_input.npy/.mat/.png")
    print(f"  - {output_prefix}_predicted.npy/.mat/.png")


def visualize_results(input_image, predicted_image, output_prefix):
    """
    視覺化預測結果
    """
    input_2d = input_image[0, :, :, 0]
    predicted_2d = predicted_image[0, :, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 輸入影像
    im1 = axes[0].imshow(input_2d, cmap='viridis')
    axes[0].set_title('Input (Dilated Image)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 預測結果
    im2 = axes[1].imshow(predicted_2d, cmap='viridis')
    axes[1].set_title('Predicted (Deconvolved)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # 儲存圖片
    output_dir = os.path.join(OUTPUT_DIR, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  - {output_prefix}_comparison.png")

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
│    3. 數值範圍會自動進行背景減去 (-167.73)                           │
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

    # 預處理輸入影像
    print(f"處理輸入影像: {args.image}")
    input_image, original_size, resized_original = preprocess_image(args.image)
    print(f"  原始尺寸: {original_size}")
    print(f"  調整後尺寸: {TARGET_SIZE}")
    print(f"  數值範圍: [{input_image.min():.2f}, {input_image.max():.2f}]")
    print()

    # 進行預測
    predicted_image = predict(model, input_image)
    print(f"✓ 預測完成！")
    print(f"  輸出數值範圍: [{predicted_image.min():.2f}, {predicted_image.max():.2f}]\n")

    # 儲存結果
    save_results(input_image, predicted_image, args.output)

    # 視覺化結果
    if not args.no_display:
        visualize_results(input_image, predicted_image, args.output)

    print("\n✓ 預測完成！")


if __name__ == '__main__':
    main()
