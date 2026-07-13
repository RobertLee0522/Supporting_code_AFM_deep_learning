# AFM 探針去卷積系統

用電腦視覺 + AFM 物理，對掃描影像做**探針去卷積**，還原樣品真實幾何。
本倉庫含**兩套獨立方法**，依需求擇一使用。

> ⚠️ **所有指令都從本專案根目錄執行**（也就是這個 README 所在的資料夾）。
> 相對路徑（`runs/`、`tip.mat`、`blind_out/` 等）都以此為基準。
> 用有裝好套件的直譯器（本機為 `py`；若你的環境是 `python` 就自行替換）。

---

## 🚀 先看這裡：我該執行哪一個？

### 方法 A ── 零訓練去卷積（**推薦、免訓練、可直接用**）
`zerotrain/` 資料夾。基於 Villarrubia 形態學，探針當「執行時輸入」，換任何探針都適用，
數學保證不過度去卷積，並誠實標出探針碰不到的死角。

| 我想做的事 | 執行 |
|-----------|------|
| **開圖形介面回推（主力）** | **雙擊 `啟動去卷積工具.bat`**，或 `py zerotrain/blind_deconvolution.py` |
| 命令列去卷積（廠商 cone） | `py zerotrain/blind_deconvolution.py 你的檔.000 --cone-R 8 --cone-theta 20.7 --sample bump` |
| 合成資料驗證 + 教學圖 | `py zerotrain/blind_deconvolution.py --demo` |
| 凸起樣品逐線 + 3D 重建 | `py zerotrain/reconstruct_lines_3d.py 你的檔.000 --R 8 --theta 20.7` |
| 診斷 Nanoscope 檔頭 | `py zerotrain/inspect_header.py 你的檔.000` |

GUI 內建三分頁：**Section 量測**（仿 NanoScope，拉線＋游標量寬度）、
**去卷積詳情**（輸入/還原/certainty/探針形狀）、**3D 還原**（可旋轉、可調 Z 軸高度）。

### 方法 B ── 深度學習 pipeline（研究用，**需先訓練**）
`deeplearning/` 資料夾。為「單一探針 + 單一樣品」訓練 U-Net。四步驟依序執行：

| 步驟 | 執行 | 產出 |
|------|------|------|
| ① 盲探針重建 GUI | `py deeplearning/afm_gui.py` | `tip.mat/tip_estimated.mat` |
| ② 探針校正（可選） | `py deeplearning/tip_correction.py tip.mat/tip_estimated.mat --size 55` | `*_corrected.mat` |
| ③ 訓練 U-Net | `py deeplearning/Supporting_code_AFM_deep_learning.py` | `runs/train{N}/weights/*.keras` |
| ④ 推論 | `py deeplearning/detect.py datafortip/std.004 -o result_std` | `img/predictions/` |
| ④b 批次推論 | `py deeplearning/batch_detect.py 資料夾/ --pattern "*.004" --output-dir results/` | 多檔結果 |
| 輔助：模型評估 | `py deeplearning/evaluate.py -m runs/train{N}/weights/AFM_MAE_autoencoder.keras` | 指標/曲線 |
| 輔助：輸入檢查 | `py deeplearning/check_image.py 你的檔` | 格式報告 |

> **不確定用哪個？** 先用 **方法 A**（免訓練、立即可跑）。方法 B 只有在你要為特定
> 探針/樣品訓練專用模型時才需要，且目前 cone 探針的訓練資料改動**尚未重跑**（見 CLAUDE.md）。

---

## 📁 資料夾結構

```
afm/                                  ← 專案根目錄（從這裡執行所有指令）
├── 啟動去卷積工具.bat                 ← 雙擊開零訓練 GUI
├── README.md / CLAUDE.md             ← 說明 / 開發規範
├── requirements.txt
│
├── zerotrain/                        ← 方法 A：零訓練去卷積（推薦）
│   ├── blind_deconvolution.py        ·  主程式 + GUI
│   ├── reconstruct_lines_3d.py       ·  凸起樣品 3D 重建
│   └── inspect_header.py             ·  Nanoscope 檔頭診斷
│
├── deeplearning/                     ← 方法 B：深度學習 pipeline（需訓練）
│   ├── afm_gui.py                    ·  ① 盲探針重建
│   ├── tip_correction.py             ·  ② 探針校正
│   ├── Supporting_code_AFM_deep_learning.py  ·  ③ 訓練
│   ├── detect.py / batch_detect.py   ·  ④ 推論 / 批次
│   ├── evaluate.py / check_image.py  ·  輔助工具
│
├── docs/                             ← 補充文件
│   ├── DETECT_README.md · GPU_SETUP_GUIDE.md · OPTIMIZATION_SUMMARY.md
│
├── lab_notebook/                     ← 電子研究紀錄簿
│
│  以下為資料/產出（未進版控，由程式讀寫，維持在根目錄）：
├── datafortip/    ← 輸入掃描檔（std.004、01-500k.tif …）
├── tip.mat/       ← 重建/校正探針
├── runs/          ← 訓練紀錄（train{N} 自動遞增）
├── img/           ← DL 中間資料 / predictions/
├── blind_out/ · blind_demo/ · recon3d_out/  ← 零訓練輸出
└── afm/           ← Python 虛擬環境（勿動）
```

---

## 安裝

```bash
pip install -r requirements.txt
```
Python 3.8+（建議 3.9–3.10）。方法 A 為 **TF-free**，只需 numpy/scipy/matplotlib；
方法 B 需 TensorFlow（GPU 設定見 `docs/GPU_SETUP_GUIDE.md`）。

---

## 物理原理（重點）

- **Grey Dilation/Erosion 對偶**：AFM 成像＝表面 ⊕ 探針（膨脹）；去卷積＝掃描 ⊖ 探針（腐蝕，Villarrubia 1997）。
- **凸起 vs 凹孔**：凸起頂端探針爬得到、只有側壁被撐寬 → erosion 還原得好；
  凹孔底部探針進不去、資訊物理性遺失 → 只能給「確定下界」並標死角。
- **零訓練的優勢**：`s ≤ s_r ≤ i` 數學成立，不會過度去卷積；certainty map 誠實標出碰不到的區域。
- **只用孔洞訓練（方法 B）**：孔洞 dilation 變淺變窄、凸起變寬，方向相反，混合會混淆。

各步驟細節與 FAQ 見 `docs/DETECT_README.md`、`docs/OPTIMIZATION_SUMMARY.md` 與 `CLAUDE.md`。

---

## Citation

```
Villarrubia, J.S. (1997). Algorithms for Scanned Probe Microscope Image Simulation,
Surface Reconstruction, and Tip Estimation. J. Res. NIST, 102(4), 425-454.
```
