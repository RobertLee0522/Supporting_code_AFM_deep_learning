#!/usr/bin/env python3
"""
批次處理 AFM 影像預測
可以處理整個資料夾中的所有影像
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path


def batch_predict(input_dir, pattern='*.tif *.000 *.001 *.002 *.003 *.004', output_dir=None):
    """
    批次處理資料夾中的所有影像

    Args:
        input_dir: 輸入資料夾路徑
        pattern: 檔案匹配模式 (預設: *.tif)
        output_dir: 輸出資料夾 (可選)
    """
    # 尋找所有匹配的檔案（支援多個 pattern，以空白分隔）
    files = []
    for pat in pattern.split():
        search_pattern = os.path.join(input_dir, '**', pat)
        files.extend(glob.glob(search_pattern, recursive=True))
    files = sorted(set(files))

    if not files:
        print(f"找不到符合 '{pattern}' 的檔案於 '{input_dir}'")
        return

    print(f"找到 {len(files)} 個檔案")
    print("="*70)

    # 處理每個檔案
    for i, file_path in enumerate(files, 1):
        print(f"\n處理 [{i}/{len(files)}]: {file_path}")
        print("-"*70)

        # 產生輸出檔名前綴
        file_stem = Path(file_path).stem
        if output_dir:
            output_prefix = os.path.join(output_dir, file_stem)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_prefix = file_stem

        # 執行 detect.py
        cmd = [
            sys.executable, 'detect.py',
            file_path,
            '-o', output_prefix,
            '--no-display'  # 批次模式不顯示圖片
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"錯誤處理 {file_path}: {e}")
            continue

    print("\n" + "="*70)
    print(f"批次處理完成！共處理 {len(files)} 個檔案")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='批次處理 AFM 影像')
    parser.add_argument('input_dir', help='輸入資料夾路徑')
    parser.add_argument('--pattern', '-p', default='*.tif', help='檔案匹配模式 (預設: *.tif)')
    parser.add_argument('--output-dir', '-o', help='輸出資料夾 (可選)')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"錯誤：找不到資料夾 '{args.input_dir}'")
        return

    batch_predict(args.input_dir, args.pattern, args.output_dir)


if __name__ == '__main__':
    main()
