#!/usr/bin/env python3
"""
診斷工具：印出 Nanoscope 原始檔標頭中每個 channel 的
@2:Z scale 原始文字、Bytes/pixel、Data offset，並讀取該 channel
的原始整數資料統計（min/max/std），用於排查：
  1) z_lsb 解析錯誤（如某些 channel 標頭格式與一般 channel 不同）
  2) bpp（dtype 寬度）解析錯誤導致 raw 值被誤判
  3) raw 資料本身變異過小（樣品太平或對焦/掃描異常）

用法：
    python inspect_header.py <檔案路徑>
"""
import sys
import re
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("用法: python inspect_header.py <檔案路徑>")
        return

    fp = sys.argv[1]
    with open(fp, 'rb') as f:
        raw_hdr = f.read(65536)
    hdr = raw_hdr.decode('latin-1')

    print("="*70)
    print(f"檔案: {fp}")
    print("="*70)

    n_px = re.search(r'\\Samps/line:\s*(\d+)', hdr)
    n_ln = re.search(r'\\(?:Number of lines|Lines):\s*(\d+)', hdr)
    n_px = int(n_px.group(1)) if n_px else 256
    n_ln = int(n_ln.group(1)) if n_ln else 256
    print(f"  Samps/line={n_px}  Lines={n_ln}")

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

        off = re.search(r'\\Data offset:\s*(\d+)', blk)
        bpp = re.search(r'\\Bytes/pixel:\s*(\d+)', blk)
        ln  = re.search(r'\\Data length:\s*(\d+)', blk)
        print(f"  Data offset : {off.group(1) if off else 'NOT FOUND'}")
        print(f"  Bytes/pixel : {bpp.group(1) if bpp else 'NOT FOUND (預設用 2)'}")
        print(f"  Data length : {ln.group(1) if ln else 'NOT FOUND'}")

        zline = re.search(r'@2:Z scale:.*', blk)
        print(f"  原始 Z scale 行: {zline.group(0) if zline else 'NOT FOUND'}")

        zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
        if zsc:
            print(f"  解析出 z_key='{zsc.group(1)}'  z_lsb={zsc.group(2)}")
        else:
            print(f"  ⚠ 目前 regex 無法解析此行（z_key/z_lsb 將使用預設值）")

        # 讀取該 channel 原始整數資料統計（未限制 data_length，舊版行為，
        # 可能讀到下一個 channel 的資料區）
        if off and bpp:
            offset_v = int(off.group(1))
            bpp_v    = int(bpp.group(1))
            dt = np.int16 if bpp_v == 2 else np.int32
            n  = n_px * n_ln
            with open(fp, 'rb') as f:
                f.seek(offset_v)
                raw_data = np.frombuffer(f.read(n * bpp_v), dtype=dt)
            if raw_data.size > 0:
                print(f"  [未限制長度] 讀到元素數: {raw_data.size}  dtype={dt.__name__}")
                print(f"  [未限制長度] raw min/max: {raw_data.min()} / {raw_data.max()}"
                      f"  std={raw_data.astype(np.float64).std():.2f}")
            else:
                print(f"  ⚠ 讀不到任何資料（offset/bpp 可能有誤）")

            # 修正後：以 Data length 為界，避免跨 channel 污染
            if ln:
                bound_bytes = int(ln.group(1))
                with open(fp, 'rb') as f:
                    f.seek(offset_v)
                    raw_bound = np.frombuffer(f.read(bound_bytes), dtype=dt)
                actual_lines = raw_bound.size // n_px
                print(f"  [Data length限制] 讀到元素數: {raw_bound.size}  "
                      f"→ {actual_lines}/{n_ln} 行有效")
                if raw_bound.size > 0:
                    print(f"  [Data length限制] raw min/max: "
                          f"{raw_bound.min()} / {raw_bound.max()}"
                          f"  std={raw_bound.astype(np.float64).std():.2f}")
        print()


if __name__ == '__main__':
    main()
