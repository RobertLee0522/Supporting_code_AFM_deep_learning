# AFM Deep Learning — 探針去卷積系統

利用卷積自編碼器（Convolutional Autoencoder）對 AFM 掃描影像進行探針去卷積，
還原凹洞型樣品（梯形孔、圓柱孔）的真實幾何形貌。

---

## 完整工作流程

```
┌──────────────────────────────────────────────────────────────────┐
│  你的輸入檔案                                                      │
│  ├── std.004          ← AFM 掃描原始檔（Nanoscope 格式）           │
│  └── 01-500k.tif      ← SEM 影像（量測幾何尺寸用）                  │
└─────────────────┬────────────────────────────────────────────────┘
                  │
         Step 1  ▼
┌─────────────────────────────┐
│  afm_gui.py                 │   GUI 探針重建
│  輸入：std.004 + SEM 幾何   │──→  tip.mat/tip_estimated.mat
└─────────────────────────────┘
                  │
         Step 2  ▼  (可選診斷)
┌─────────────────────────────┐
│  tip_correction.py          │   探針方向校正
│  輸入：tip_estimated.mat    │──→  tip_estimated_corrected.mat
└─────────────────────────────┘
                  │
         Step 3  ▼
┌─────────────────────────────┐
│  Supporting_code_AFM_       │   神經網路訓練
│  deep_learning.py           │──→  runs/train{N}/weights/*.keras
│  （合成孔洞資料生成 + 訓練）  │
└─────────────────────────────┘
                  │
         Step 4  ▼
┌─────────────────────────────┐
│  detect.py                  │   去卷積推論
│  輸入：任意 AFM 掃描檔       │──→  img/predictions/
└─────────────────────────────┘
```

---

## 各步驟詳細說明

### Step 1：探針重建（`afm_gui.py`）

使用 AFM 掃描的校正樣品（已知幾何）透過 Villarrubia 盲探針算法重建探針形狀。

```bash
python afm_gui.py
```

**GUI 操作流程：**
1. 點選「瀏覽」→ 選擇 `datafortip/std.004`
2. 選擇高度通道（通常是 Height Sensor）
3. 特徵形狀選「梯形孔 Trapezoid Hole」
4. 輸入 SEM 量測幾何（來自 `datafortip/01-500k.tif`）：

| 參數 | 數值 |
|------|------|
| 開口寬 d_top | 228.8 nm |
| 底部寬 d_bot | 183.5 nm |
| 深度 depth   | 125.8 nm |
| 週期 pitch   | 依 SEM 量測（典型 ~1000 nm）|

5. 點選「▶ 執行重建」
6. 確認 Cross-section Profile 合理後，點選「💾 儲存 tip.mat」
7. 儲存路徑建議：`tip.mat/tip_estimated.mat`

> **進入率 < 50%** 時表示探針無法完全深入孔洞，此時 Rim 分析更可靠。

---

### Step 2：探針校正（`tip_correction.py`）— 可選診斷

檢查重建探針的品質，校正方向偏移、強制旋轉對稱。

```bash
python tip_correction.py tip.mat/tip_estimated.mat --size 55
```

輸出：
- `tip.mat/tip_estimated_corrected.mat`  ← 校正後的探針
- `tip.mat/tip_estimated_correction_report.png`  ← 四格診斷圖

| 參數 | 說明 |
|------|------|
| `--size 55` | 輸出探針尺寸（預設 55×55 px）|
| `--no-display` | 不彈出視窗，僅存圖 |
| `--diagnose-only` | 只診斷，不儲存 |

---

### Step 3：訓練模型（`Supporting_code_AFM_deep_learning.py`）

自動生成合成孔洞資料（梯形孔 + 圓柱孔），訓練卷積自編碼器。

```bash
python Supporting_code_AFM_deep_learning.py
```

**訓練設定（程式碼頂端常數）：**

| 常數 | 預設值 | 說明 |
|------|--------|------|
| `TIP_RADIUS_NM` | 73.0 nm | 合成訓練探針的曲率半徑（從 tip_estimated0526.mat 量得）|
| `TRAP_OPEN_NM`  | 228.8 nm | 梯形孔開口寬度（SEM 量測）|
| `TRAP_BOT_NM`   | 183.5 nm | 梯形孔底部寬度（SEM 量測）|
| `TRAP_DEPTH_NM` | 125.8 nm | 梯形孔深度（SEM 量測）|
| `NORM_MIN`      | -155 nm  | 全域正規化下限 |
| `NORM_MAX`      | +105 nm  | 全域正規化上限 |
| `EPOCHS`        | 2000     | 訓練輪數 |

**訓練輸出（自動存至 `runs/train{N}/`）：**
```
runs/
└── train4/                          ← 本次訓練目錄（自動遞增）
    ├── weights/
    │   ├── AFM_MAE_autoencoder.keras
    │   └── AFM_MAE_autoencoder.weights.h5
    ├── 01_cylinder_hole_preview.png  ← 圓柱孔 ground-truth 預覽
    ├── 02_trapezoid_preview.png      ← 梯形孔 ground-truth 預覽
    ├── 03_tip_shape.png              ← 訓練用合成探針形狀
    ├── 04_trap_dilation_check.png    ← 孔洞 dilation 前後比較
    ├── 05_training_samples.png       ← 訓練資料樣本
    ├── 06_loss_curve.png             ← 損失曲線
    ├── 07_predictions_grid.png       ← 預測結果矩陣
    ├── 08_triplet_sample_0.png       ← 輸入/GT/預測三圖對比
    ├── 09_profile_sample_0.png       ← 剖面比較圖
    ├── metrics.txt                   ← 評估指標（MAE/RMSE/SSIM/PSNR）
    └── config.txt                    ← 訓練設定記錄
```

---

### Step 4：推論（`detect.py`）

對新的 AFM 掃描做去卷積。**直接支援 Nanoscope 原始格式（`.000`–`.009`）**。

```bash
# 使用 Nanoscope 原始掃描檔
python detect.py datafortip/std.004 -o result_std

# 使用一般影像（需為 nm 單位的高度影像）
python detect.py my_scan.tif -o my_result

# 指定模型路徑
python detect.py std.004 -m runs/train4/weights/AFM_MAE_autoencoder.keras -o result

# 批次處理整個資料夾
python batch_detect.py my_afm_folder/ --pattern "*.004" --output-dir results/
```

**輸出（`img/predictions/`）：**
```
img/predictions/
├── result_std_input.npy      ← 輸入（nm）
├── result_std_predicted.npy  ← 預測（nm）
├── result_std_input.mat      ← MATLAB 格式
├── result_std_predicted.mat  ← MATLAB 格式
├── result_std_input.png      ← 視覺化
├── result_std_predicted.png  ← 視覺化
└── result_std_comparison.png ← 輸入/預測並排比較
```

---

## 快速開始（一鍵指令）

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 探針重建（GUI）
python afm_gui.py

# 3. 訓練（約 30–60 分鐘，GPU 加速）
python Supporting_code_AFM_deep_learning.py

# 4. 推論
python detect.py datafortip/std.004 -o my_result
```

---

## 檔案結構

```
afm/
├── afm_gui.py                          # Step 1：探針重建 GUI
├── tip_correction.py                   # Step 2：探針校正（可選）
├── Supporting_code_AFM_deep_learning.py # Step 3：訓練
├── detect.py                           # Step 4：推論
├── batch_detect.py                     # Step 4b：批次推論
├── evaluate.py                         # 模型評估工具
├── requirements.txt                    # Python 套件需求
│
├── datafortip/
│   ├── std.004                         # ← 你的 AFM 掃描原始檔
│   └── 01-500k.tif                     # ← SEM 影像（幾何量測）
│
├── tip.mat/
│   ├── tip_estimated.mat               # afm_gui.py 輸出的估算探針
│   └── tip_estimated_corrected.mat     # tip_correction.py 校正後
│
├── img/                                # 中間資料（訓練/推論）
│   └── predictions/                    # 推論輸出
│
└── runs/                               # 訓練紀錄（YOLO 風格自動遞增）
    ├── train1/
    ├── train2/
    └── train{N}/                       ← 最新訓練
        ├── weights/
        ├── *.png
        ├── metrics.txt
        └── config.txt
```

---

## 系統需求

| 項目 | 最低 | 建議 |
|------|------|------|
| Python | 3.8 | 3.9–3.10 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 不需要 | 6 GB（NVIDIA）|
| 訓練時間 | ~60 min (CPU) | ~10 min (GPU) |

```bash
pip install -r requirements.txt
```

**GPU 加速（可選）：**
詳見 `GPU_SETUP_GUIDE.md`

---

## 物理原理說明

### 為何只用孔洞資料訓練？

AFM Grey Dilation：`dilated[i,j] = max_{u,v}{ surface[i-u,j-v] + tip[u,v] }`

| 樣品類型 | Dilation 效果 | 模型需學到 |
|----------|--------------|-----------|
| 凸起突粒 | 突起**變寬**（擴散） | 縮窄 |
| **凹入孔洞** | 孔洞**變淺變窄**（探針進不去）| **加深、加寬** |

這兩種操作**方向完全相反**，混合訓練會讓模型混淆。由於你的樣品是孔洞，訓練資料全部使用孔洞。

### 全域正規化的作用

| 數值 | 原始 nm | 正規化後 |
|------|---------|---------|
| 平坦表面 | 0 nm | **0.597**（非零！）|
| 孔洞底部 | -125 nm | 0.115 |
| 訓練前    | — | 模型輸出全零（最省力解）|
| 正規化後  | — | 模型必須學習真實結構 |

正規化把「背景 = 0」這個特殊值消除，避免模型找到「全輸出零」的偷懶解。

---

## 探針重建注意事項

### 關於 afm_gui.py 的「抹平」處理
GUI 對估算探針進行三層處理：
1. Villarrubia 形態學腐蝕
2. 四向對稱化（X/Y 各翻轉取平均）
3. **徑向平均（抹平）** ← 強制旋轉對稱

這是**有損**處理，假設探針是旋轉對稱的。對訓練無影響（訓練用合成拋物線探針已完美對稱）。

### 進入率（Entry Rate）
若 Entry Rate < 50%，代表探針沒有完全到達孔底：
- 孔洞較深或較窄時常見
- 此時 **Rim Analysis** 比 Tip 3D 更可靠
- 可用`壁角 = arctan((開口-底部)/(2×深度))`交叉驗證

---

## 常見問題

**Q: 模型預測輸出全黑（全零）？**
→ 確認訓練時有套用正規化（`NORM_MIN=-155`, `NORM_MAX=+105`），且模型最後一層使用 `sigmoid` 激活函式。

**Q: detect.py 讀不了 .004 檔？**
→ 確認副檔名為純 3 位數字（`.000`~`.009`），程式會自動辨識 Nanoscope 格式。

**Q: 訓練資料要多少？**
→ 預設梯形孔 400 張 + 圓柱孔 400 張 = 800 張，train:test = 8:2。
可修改 `N = 400` 增加資料量（建議最多 1000 張）。

**Q: 如何評估訓練品質？**
→ 看 `runs/train{N}/metrics.txt`：
- SSIM > 0.85：良好
- MAE < 5 nm：優秀
- 看 `04_trap_dilation_check.png`：孔洞在 dilation 後應變淺/變窄

---

## 引用 / Citation

本程式基於 Villarrubia (1997) 盲探針重建算法，
並結合卷積自編碼器進行深度學習去卷積。

```
Villarrubia, J.S. (1997). Algorithms for Scanned Probe Microscope 
Image Simulation, Surface Reconstruction, and Tip Estimation. 
Journal of Research of NIST, 102(4), 425-454.
```
