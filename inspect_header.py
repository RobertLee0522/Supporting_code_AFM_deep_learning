#!/usr/bin/env python3
"""
診斷工具：印出 Nanoscope 原始檔標頭中每個 channel 的
@2:Z scale 原始文字與其他關鍵欄位，用於排查 z_lsb 解析錯誤
（例如某些 Height Sensor channel 標頭格式與一般 channel 不同，
導致 z_lsb 被解析成異常小的數值，造成影像被誤縮放成近乎全 0）。

用法：
    python inspect_header.py <檔案路徑>
"""
import sys
import re


def main():
    if len(sys.argv) != 2:
        print("用法: python inspect_header.py <檔案路徑>")
        return

    fp = sys.argv[1]
    with open(fp, 'rb') as f:
        raw = f.read(65536)
    hdr = raw.decode('latin-1')

    print("="*70)
    print(f"檔案: {fp}")
    print("="*70)

    # 全域 Z 靈敏度欄位
    for pat, label in [
        (r'@Sens\. ZsensSens:\s*V\s*([\d.]+)', 'ZsensSens (global)'),
        (r'@Sens\. Zsens:\s*V\s*([\d.]+)',     'Zsens (global)'),
    ]:
        m = re.search(pat, hdr)
        print(f"  {label}: {m.group(0) if m else 'NOT FOUND'}")

    print()
    blocks = hdr.split('\\*Ciao image list')
    for i, blk in enumerate(blocks):
        if not blk.strip():
            continue
        nm = re.search(r'@2:Image Data:\s*S\s*\[([^\]]+)\].*?"([^"]+)"', blk)
        if not nm:
            continue
        name = nm.group(2).strip()
        print(f"--- Channel #{i}: {name} ---")

        zline = re.search(r'@2:Z scale:.*', blk)
        print(f"  原始 Z scale 行: {zline.group(0) if zline else 'NOT FOUND'}")

        zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
        if zsc:
            print(f"  解析出 z_key='{zsc.group(1)}'  z_lsb={zsc.group(2)}")
        else:
            print(f"  ⚠ 目前 regex 無法解析此行（z_key/z_lsb 將使用預設值）")
        print()


if __name__ == '__main__':
    main()
