# Detect.py 使用說明

## 快速開始

### 基本用法
```bash
python detect.py path/to/your/image.tif
```

### 範例（使用您提到的檔案）
```bash
python detect.py "C:\Users\54-0461100-01\Desktop\python\FW115\afm\分析\AD-40-AS.tif"
```

### 顯示輸入要求說明
```bash
python detect.py --info
```

## 輸入影像格式要求

### 什麼是「膨脹影像」(Dilated Image)？

```
真實表面形貌 (Ground Truth)          AFM 掃描影像 (Dilated Image - 輸入)
        ↓                                      ↓
    ┌─────┐                              ┌─────┐
    │  ╱╲  │      AFM 探針掃描            │  ╱╲_│
    │ ╱  ╲ │    ───────────────────>    │ ╱   │
    │╱    ╲│                              │╱    │
    └─────┘                              └─────┘
    (尖銳特徵)                           (被探針半徑膨脹)
```

**模型輸入**：經過 AFM 探針卷積後的「膨脹影像」  
**模型輸出**：還原後的真實表面形貌

### 支援格式
- `.tif`, `.tiff` - TIFF 格式（推薦）
- `.png` - PNG 格式
- `.jpg`, `.jpeg` - JPEG 格式
- `.bmp` - BMP 格式

### 影像前處理流程

1. **尺寸調整**：自動調整為 128×128 像素
2. **灰階轉換**：彩色影像自動轉為灰階
3. **背景減去**：減去 167.73（與訓練時相同）

### 數值單位

- 輸入影像的像素值應該代表**高度資訊**（單位：nm 或 Å）
- 模型會輸出去卷積後的高度圖

## 輸出檔案

執行後會在 `img/predictions/` 資料夾產生：

```
img/predictions/
├── {prefix}_input.npy         # 輸入影像 (NumPy 格式)
├── {prefix}_input.mat         # 輸入影像 (MATLAB 格式)
├── {prefix}_input.png         # 輸入影像 (影像格式)
├── {prefix}_predicted.npy     # 預測結果 (NumPy 格式)
├── {prefix}_predicted.mat     # 預測結果 (MATLAB 格式)
├── {prefix}_predicted.png     # 預測結果 (影像格式)
└── {prefix}_comparison.png    # 對比圖
```

## 完整範例

### 單張影像預測
```bash
# 基本預測
python detect.py "C:\Users\54-0461100-01\Desktop\python\FW115\afm\分析\AD-40-AS.tif"

# 指定輸出檔名前綴
python detect.py "C:\Users\54-0461100-01\Desktop\python\FW115\afm\分析\AD-40-AS.tif" -o "AD40_result"

# 不顯示圖片（僅儲存）
python detect.py "analysis\AD-40-AS.tif" --no-display
```

### 批次處理多張影像
```bash
# Windows (PowerShell)
Get-ChildItem "C:\path\to\images\*.tif" | ForEach-Object { python detect.py $_.FullName -o $_.BaseName }

# Windows (CMD)
for %f in (C:\path\to\images\*.tif) do python detect.py "%f" -o "%~nf"
```

## 常見問題

### Q: 我的影像看起來很模糊，預測結果不理想？
**A**: 請確認輸入影像是「膨脹影像」而非已經處理過的影像。模型設計用於去除 AFM 探針造成的卷積效應。

### Q: 可以使用原始 AFM 儀器輸出的檔案嗎？
**A**: 可以，只要將檔案轉換為支援的影像格式即可。注意某些 AFM 格式可能需要先轉換。

### Q: 影像尺寸不是 128×128 會怎樣？
**A**: 會自動調整為 128×128。如果原始解析度不同，可能會影響預測品質。

### Q: 為什麼要做背景減去 (-167.73)？
**A**: 這是與訓練資料前處理一致的步驟，用於修正 `scipy.ndimage.grey_dilation` 造成的數值偏移。

## 注意事項

⚠️ **重要**：輸入影像必須是 AFM 掃描後的原始「膨脹影像」，而非：
- ❌ 已經去卷積處理過的影像
- ❌ 經過其他濾波處理的影像
- ❌ 只有部分區域的裁切影像（除非裁切後仍為 128×128）

✅ 正確的輸入應該顯示出探針造成的「膨脹」效應，特別是在尖銳特徵周圍。
