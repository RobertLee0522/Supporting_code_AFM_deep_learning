# TensorFlow GPU 設定指南 (AFM 專案專用)

## 當前系統檢查 (2026-04-27)
*   **偵測到系統 CUDA**: 12.1 (透過 `nvcc` 驗證)
*   **專案需求版本**: 11.2 (對應 TensorFlow 2.10.1)
*   **當前狀態**: 軟體套件已於 `afm` 環境中修復，但因 CUDA 版本過新，目前僅能以 CPU 執行。

## 安全性提醒：會影響其他環境嗎？
**不會。** 
1.  **Conda 隔離**：本指南建議的所有 `pip` 操作僅限於 `afm` 環境內，不會影響 `cellpose`、`sam2` 等其他 Conda 環境。
2.  **CUDA 並存**：Windows 支援多個 CUDA 版本並存。安裝 11.2 **不需要** 移除 12.1，兩者可同時存在於硬碟中。

---

## 解決方案：安裝 CUDA 11.2 (啟用 GPU 加速)

為了讓 AFM 專案偵測到 GPU，請依照以下步驟操作：

### 步驟 1: 下載與安裝 CUDA 11.2
1.  前往 [NVIDIA CUDA 11.2 下載頁面](https://developer.nvidia.com/cuda-11.2.0-download-archive)。
2.  安裝時選擇**「自定義」**，確保不會複寫掉現有的驅動程式（如果要保留 12.1）。
3.  預設路徑應為：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`。

### 步驟 2: 安裝 cuDNN 8.1
1.  下載 cuDNN 8.1 for CUDA 11.x。
2.  解壓縮後，將 `bin`, `include`, `lib` 資料夾內的檔案**複製**到 CUDA 11.2 的對應路徑。

### 步驟 3: 設定環境變數 (關鍵要點)
為了確保不干擾系統其他程式，建議將 11.2 的路徑加入**「使用者變數」**而非「系統變數」，或在啟動程式前透過腳本指定：
*   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
*   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

---

## 快速檢查清單 (軟體層面已自動修復)
在 `afm` 環境中，我已為您完成以下配置：
- [x] Python 3.9
- [x] TensorFlow 2.10.1 (Windows GPU 最後版本)
- [x] NumPy < 2.0.0 (避免不相容崩潰)
- [x] 其他必要套件 (opencv, scipy, etc.)

## 驗證 GPU 指令
在 Conda 的 `afm` 環境下執行：
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 常見問題

### Q: 為什麼不直接升級 TensorFlow 到最新版？
**A**: TensorFlow 2.11 之後的版本在 Windows 上**原生不支援 GPU**。若要使用最新版 GPU，必須切換到 WSL2 (Linux 模式)，這對檔案存取較不方便。因此 2.10.1 是 Windows 原生環境的最佳選擇。

### Q: 如果我之後要跑 YOLO 或 SAM2 怎麼辦？
**A**: 直接切換回原本的環境 (如 `conda activate sam2`)。那些環境會自動去找與它們相容的 CUDA 12.1，不會被這邊的 11.2 干擾。
