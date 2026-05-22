#!/usr/bin/env python3
"""
檢查影像格式是否符合 detect.py 要求
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def check_image(image_path):
    """檢查影像格式。"""
    print("="*70)
    print(f"檢查影像: {image_path}")
    print("="*70)

    # 檢查檔案是否存在
    if not os.path.exists(image_path):
        print(f"❌ 錯誤：檔案不存在")
        return False

    # 嘗試開啟影像
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"❌ 錯誤：無法開啟影像 - {e}")
        return False

    print(f"✓ 成功開啟影像")
    print(f"\n基本資訊:")
    print(f"  格式: {img.format}")
    print(f"  模式: {img.mode}")
    print(f"  尺寸: {img.size}")
    print(f"  檔案大小: {os.path.getsize(image_path) / 1024:.1f} KB")

    # 轉換為 numpy array
    img_array = np.array(img)
    print(f"\n數值資訊:")
    print(f"  Array shape: {img_array.shape}")
    print(f"  資料類型: {img_array.dtype}")
    print(f"  數值範圍: [{img_array.min()}, {img_array.max()}]")
    print(f"  平均值: {img_array.mean():.2f}")
    print(f"  標準差: {img_array.std():.2f}")

    # 檢查是否為多層 TIFF
    if img.format == 'TIFF' and len(img_array.shape) == 3:
        print(f"  ⚠️ 注意：這是多層 TIFF (pages={img_array.shape[0]})")
        print(f"    將使用第一層進行處理")

    # 顯示建議
    print(f"\n檢查結果:")

    # 檢查影像模式
    if img.mode == 'RGB' or img.mode == 'RGBA':
        print(f"  ⚠️ 彩色影像將自動轉換為灰階")
    elif img.mode == 'L':
        print(f"  ✓ 已是灰階影像")
    else:
        print(f"  ⚠️ 不常見的影像模式: {img.mode}")

    # 檢查尺寸
    if img.size == (128, 128):
        print(f"  ✓ 尺寸正好為 128×128")
    else:
        print(f"  ℹ️  尺寸將調整為 128×128 (原始: {img.size})")

    # 顯示預覽
    print(f"\n顯示影像預覽...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 原始影像
    if len(img_array.shape) == 3 and img_array.shape[0] < 10:
        # 多層影像，顯示第一層
        display_img = img_array[0] if len(img_array.shape) == 3 else img_array
    else:
        display_img = img_array

    if len(display_img.shape) == 3:
        display_img = display_img[:, :, 0] if display_img.shape[2] < 5 else np.mean(display_img, axis=2)

    im1 = axes[0].imshow(display_img, cmap='viridis')
    axes[0].set_title(f'Original ({img.size[0]}×{img.size[1]})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 調整尺寸後
    img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
    if img_resized.mode != 'L':
        img_resized = img_resized.convert('L')
    resized_array = np.array(img_resized)

    im2 = axes[1].imshow(resized_array, cmap='viridis')
    axes[1].set_title('Resized (128×128)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    print(f"\n✓ 影像檢查完成，可以使用 detect.py 進行預測")
    print(f"  建議指令: python detect.py \"{image_path}\"")
    print("="*70)

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方式: python check_image.py <image_path>")
        print("\n範例:")
        print('  python check_image.py "C:\\path\\to\\image.tif"')
        sys.exit(1)

    image_path = sys.argv[1]
    check_image(image_path)
