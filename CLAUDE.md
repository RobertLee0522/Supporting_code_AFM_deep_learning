# CLAUDE.md

本檔案是給 Claude Code（及任何協作者）閱讀的專案說明書，描述
**AFM 探針去卷積（Tip Deconvolution）系統**的整體邏輯、程式架構、
電腦視覺與 AFM 專業原理，以及本倉庫的開發規範。

> ⚠️ **第一條鐵則**：每次更新程式碼或結構，**都必須同步修改本 `CLAUDE.md`**
> （至少在文末「變更紀錄 Changelog」追加一筆）。詳見 [§7 倉庫規範](#7-倉庫規範-repository-rules)。

---

## 1. 專案總覽 (Overview)

本專案結合 **電腦視覺 (Computer Vision)** 與 **掃描探針顯微鏡 (AFM)** 物理，
用 **卷積自編碼器 (Convolutional Autoencoder)** 對 AFM 掃描影像做
**探針去卷積**，還原凹洞型樣品（梯形孔、圓柱孔）的真實幾何形貌。

核心物理：AFM 影像是「真實表面」與「探針形狀」的 **灰階膨脹 (grey dilation)**
卷積結果。有限曲率半徑的探針無法完全探入窄／深的孔洞，導致孔洞在影像中
**變淺、變窄**。本系統的任務即是學習其 **逆運算（去卷積 / 腐蝕）**。

```
真實表面 (Ground Truth)  ──grey_dilation(⊕ tip)──▶  AFM 掃描影像 (變淺變窄)
                          ◀──── 模型去卷積 (學習逆映射) ────
```

關鍵設計決策：**只用孔洞 (pit/hole) 資料訓練**。凸起特徵 dilation 後變寬、
凹洞 dilation 後變淺變窄，兩者方向相反，混合訓練會讓模型混淆；本專案樣品
為孔洞，故訓練資料全為孔洞。

---

## 2. 端到端工作流程 (Pipeline)

```
輸入：std.004 (Nanoscope AFM 原始檔)  +  01-500k.tif (SEM 影像，量幾何)
  │
  ▼  Step 1  afm_gui.py ........... GUI 盲探針重建 (Villarrubia) → tip.mat
  │
  ▼  Step 2  tip_correction.py .... 探針方向/對稱/尺寸校正 (可選) → *_corrected.mat
  │
  ▼  Step 3  Supporting_code_AFM_deep_learning.py
  │          合成孔洞資料生成 + grey dilation + 訓練自編碼器
  │          → runs/train{N}/weights/AFM_MAE_autoencoder.keras
  │
  ▼  Step 4  detect.py / batch_detect.py ... 對真實 AFM 掃描去卷積推論
  │          → runs/train{N}/predictions/predict{M}/*.{npy,mat,png}
  │
  ▼  輔助    evaluate.py ........... 測試集評估 (MAE/RMSE/PSNR/SSIM)
             check_image.py ....... 推論前輸入格式檢查
```

各步驟詳細指令與參數見 `README.md`、`DETECT_README.md`、`GPU_SETUP_GUIDE.md`、
`OPTIMIZATION_SUMMARY.md`。

---

## 3. 程式架構與模組 (Architecture)

| 檔案 | 角色 | 輸入 | 輸出 |
|------|------|------|------|
| `afm_gui.py` | Step 1：Tkinter GUI 盲探針重建 | `.000`–`.009` Nanoscope + SEM 幾何 | `tip_estimated.mat` + 報告 PNG |
| `tip_correction.py` | Step 2：探針校正（置中/徑向平均/縮放） | `tip_estimated.mat` | `*_corrected.mat/.npy` + 診斷圖 |
| `Supporting_code_AFM_deep_learning.py` | Step 3：合成資料 + 訓練 | （自動生成合成資料） | `runs/train{N}/weights/*.keras`, `metrics.txt`, `config.txt` |
| `detect.py` | Step 4：單張推論 | 真實 AFM 掃描 | 去卷積結果 `.npy/.mat/.png` |
| `batch_detect.py` | Step 4b：批次推論 | 資料夾 | 多檔去卷積結果 |
| `evaluate.py` | 模型評估 | 模型 + 測試集 | 指標 CSV、loss 曲線、誤差直方圖 |
| `check_image.py` | 輸入驗證 | 單張影像 | 格式/尺寸/數值範圍報告 |
| `blind_deconvolution.py` | 通用去卷積引擎（零訓練，Villarrubia 形態學）+ **步驟式 GUI** | Nanoscope `.000` 或影像 .npy + 已知探針（對稱/**非對稱** cone 或 tip.npy） | 還原表面 + certainty map |
| `reconstruct_lines_3d.py` | 凸起樣品逐線 1D 去卷積 + 3D 重建（給定 R, θ） | Nanoscope .000 + 探針 R/θ | 還原高度圖 .npy + 3D 渲染 PNG |

### 3.1 `afm_gui.py`（盲探針重建 GUI）
- `parse_nanoscope()` / `read_channel()`：解析 Nanoscope 二進位標頭，
  讀取高度通道，套用 Z-scale（`raw × z_lsb × z_sensitivity → nm`）。
- `flatten_rows()`：逐列一階多項式去傾斜；孔洞型以 percentile(95) 為基線。
- `make_gt()` / `make_gt_patch()`：依 SEM 幾何向量化生成理想孔洞 GT。
- `reconstruct()`：偵測特徵 → 收集 patch → 平均 → 2D 配準 → 深度對齊 →
  形態學腐蝕 (`compute_tip`)。
- `compute_tip()`：**核心**。表面/基底對齊 GT 後做 Villarrubia 形態學腐蝕
  (`grey_erosion`)，再四向對稱化 + 徑向平均強制旋轉對稱，最後 `extend_tip_cone()`
  錐壁延伸。`manual_offset` 純視覺平移（見 Changelog）。
- `extend_tip_cone()`：**[FIX-11]** 把盲重建探針外緣「攤平的裙邊」沿錐壁斜率
  直線外推成 cone。盲重建只能還原探針實際接觸孔壁的頂端，接觸區外 `grey_erosion`
  的 min 飽和 → 側壁攤平變鈍；本函式量測接觸區錐壁斜率並直線延伸至邊緣（含角落），
  還原錐形側壁。下游效益：較尖錐探針 dilation 的 narrow 量正常，避免鈍探針把訓練孔
  填太窄導致模型過度去卷積。
- `reconstruct_morphological()`：`tip = scan ⊖ surface`（孔洞先反轉再腐蝕）。
- `analyze_rim()`：從孔緣過渡帶估算探針有效半徑、錐角、進入率（進入率低時
  比孔底資訊更可靠）。
- `App`（Tkinter）：側欄參數輸入 + 6 格 matplotlib 結果圖（AFM/GT/平均特徵/
  Tip 形狀/Cross-section/Rim 或 3D）。

### 3.2 `Supporting_code_AFM_deep_learning.py`（訓練核心）
- 合成 GT：`trapezoid_hole_creator/randomizer`（Chebyshev 距離梯形截錐孔）、
  `cylinder_hole_creator/randomizer`（Euclidean 距離圓柱孔，小+大混合尺度）、
  `star_hole_creator/randomizer`（極座標 `r_bound(θ)=r_mid+amp·cos(n·(θ−θ0))`
  生成 4/5/6 芒星形孔，非凸含尖角，提升形狀魯棒性）。
- 合成探針：**`make_cone_training_tip()`（廠商 cone：球冠+直線錐壁，預設 R=2nm θ=25°，
  `USE_CONE_TIP=True` 時採用）** 或 `make_training_tip()`（拋物面 `z(r)=-r²/(2R)`）或 tip.mat。
  尖 cone 對寬孔幾乎不 narrow，使訓練 input 對齊真實掃描 input（修正過度去卷積，見 Changelog）。
- 物理模擬：`scipy.ndimage.grey_dilation(surface, structure=tip)` 產生「膨脹影像」。
- 真實感：`add_scan_line_artifacts()`（橫向掃描條紋）+ `add_gaussian_noise()`。
- 正規化：全域 min/max（`NORM_MIN=-155`, `NORM_MAX=+105`）→ 背景 0nm 映射到
  ~0.597（非零），避免模型學到「全輸出 0」的偷懶解。
- 指標：`calculate_metrics()`（MAE/MSE/RMSE/PSNR/SSIM/像素準確率）。
- `TrainingProgressCallback`：每 epoch loss/ETA 顯示。

### 3.3 推論與評估
- `detect.py`：支援 Nanoscope 原生格式與一般影像；resize 至 128×128、
  正規化、推論、反正規化回 nm；Z-range 保護（超過訓練範圍 10× 自動縮放）；
  YOLO 風格自動遞增輸出目錄。
- `tip_correction.py`：`center_apex` → `radial_average` → `resize`（預設 55×55），
  強制旋轉對稱並標準化頂點為 0。

---

## 4. 神經網路架構 (Model)

**U-Net with Skip Connections**（Keras Functional API），輸入/輸出皆 `(128, 128, 1)`：

```
Encoder:  conv_block(32) → MaxPool → conv_block(64) → MaxPool
          → conv_block(128) → MaxPool          (128→64→32→16 px)
Bottleneck: conv_block(256)                    (16×16×256 = 65536 值)
Decoder:  UpConv(128)+concat(skip)+conv_block(128)  → 32×32
          UpConv(64) +concat(skip)+conv_block(64)   → 64×64
          UpConv(32) +concat(skip)+conv_block(32)   → 128×128
Output:   Conv1×1 Sigmoid                      (輸出 [0,1])
conv_block = Conv3×3 → BatchNorm → ReLU  ×2
```

| 超參數 | 值 | 理由 |
|--------|----|----|
| Optimizer | Adam(lr=3e-4) | 配合 [0,1] 正規化範圍 |
| Loss | MAE | 對離群值穩健，nm 誤差有直接物理意義 |
| Epochs (max) / Batch | 3000 / 32 | EarlyStopping(patience=400) 自動提前停止 |
| LR schedule | ReduceLROnPlateau × 0.5 (patience=100) | 避免 loss plateau |
| 輸出激活 | Sigmoid | 對應 [0,1] 正規化輸出 |
| 資料量 | 梯形孔+圓柱孔+星形孔，train:test ≈ 8:2 | 幾何多樣性提升泛化 |

---

## 5. 電腦視覺 / AFM 專業要點 (CV & AFM Notes)

- **Grey Dilation/Erosion 對偶**：AFM 成像 = 表面 ⊕ 探針（膨脹）；
  盲探針重建 = 掃描 ⊖ 表面（腐蝕，Villarrubia 1997）。
- **平移不變性陷阱**：形態學腐蝕對輸入的剛性垂直平移不變
  （`output += c` 後正規化抵消），因此「上下平移曲線」不會改變還原的 tip
  形狀；要改變 tip 必須改變曲線相對 GT 的 **形狀**（深度縮放）。
- **進入率 (Entry Rate)**：探針進入孔洞的深度比例；< 50% 時孔底資訊不可靠，
  改用 Rim 分析（孔緣過渡帶）。交叉驗證：`壁角 = arctan((開口-底部)/(2×深度))`。
- **正規化避免退化解**：把「背景=0」這個特殊值消除，否則模型會輸出全零。
- **旋轉對稱假設**：tip 校正以徑向平均強制旋轉對稱，屬有損處理；合成訓練
  探針本就對稱故無影響。
- **單位一致性**：訓練與推論的 `NORM_MIN/NORM_MAX`、px↔nm 換算必須一致，
  否則預測尺度錯誤。
- **評估指標解讀**：SSIM>0.85 良好、MAE<5nm 優秀；並檢查 dilation check 圖
  確認孔洞在膨脹後確實變淺/變窄。

---

## 6. 環境與依賴 (Environment)

```bash
pip install -r requirements.txt   # numpy scipy matplotlib Pillow tensorflow scikit-image scikit-learn ...
```
- Python 3.8+（建議 3.9–3.10）；GPU 可選（見 `GPU_SETUP_GUIDE.md`）。
- `afm_gui.py` 需 Tkinter（`matplotlib` 用 `TkAgg`）。
- **無頭測試**：以 `matplotlib.use('Agg')` 在無顯示環境驗證數值與繪圖邏輯。

---

## 7. 倉庫規範 (Repository Rules)

這些規則由 `.gitignore` 強制，協作時也務必遵守：

1. **不提交照片／圖片**：`*.png/.jpg/.tif/.pdf` 等與 `img/`、`runs/`、`結果/`、
   `分析/` 目錄一律 **不上傳**。只版控 **程式碼與輕量文字檔**
   （`*.py`、`*.md`、`requirements.txt`、`*.bat`）。
2. **不上傳大型檔案**：模型權重 `*.keras/.h5/.weights`、資料 `*.npy/.mat`、
   `*.zip/.7z/.tar.gz`、`*.ipynb` 等大檔 **不上傳**。
3. **不上傳產生物 / 暫存**：`__pycache__/`、`*.log/.tmp`、`.vscode/`、`.idea/`、
   `.DS_Store` 等。
4. **每次更新都要改本檔**：任何程式碼、架構、參數或行為變更，**必須**同步
   更新本 `CLAUDE.md`（對應章節 + 文末 Changelog 追加一筆）。送交前自我檢查：
   「這次改動是否已反映在 CLAUDE.md？」
5. **內容專業性**：文件與註解須符合 **電腦視覺與 AFM 專業**，正確使用
   dilation/erosion、去卷積、進入率、SSIM/PSNR 等術語。
6. **新增需上傳的檔案類型**：若確有必要版控新類型（如某小型參數檔），
   先在 `.gitignore` 以 `!pattern` 明確白名單，並於此說明。
   - `lab_notebook/*.md`（2026-06-16 新增白名單）：電子研究紀錄簿，
     每篇記錄一次調查/修正的目的、過程、數據與結論，輕量文字檔。

> 提交前快速檢查：`git status` 不應出現任何圖片／權重／資料大檔；
> 若出現代表 `.gitignore` 需補規則。

---

## 8. 程式開發能力與規範 (Coding Skills & Conventions)

> 本節整合「寫程式碼能力 (skill)」準則，並針對本 CV/AFM 倉庫特化。
> 開發者與 Claude Code 在本專案改動程式碼時應遵循。

### 8.1 通用工程準則
- **融入既有風格**：新程式碼的命名、註解密度、慣用法要與周圍程式碼一致
  （本倉庫慣用中文註解 + `[FIX-n]` 標記修正、區塊以 `── ... ──` 分隔）。
- **最小且聚焦的改動**：只動需要動的部分；不順手大改無關區域。
- **可逆與安全**：刪除／覆寫前先確認目標內容；破壞性或對外動作先確認。
- **忠實回報**：測試失敗就說失敗並附輸出；跳過的步驟要講明。
- **善用專用工具**：讀寫檔案用編輯工具而非 `cat/sed`；搜尋用對應工具。
- **平行化**：彼此獨立的查詢／工具呼叫一次發出以加速。

### 8.2 本專案特化準則
- **NumPy 向量化**：避免雙層 Python 迴圈（如 `make_gt` 用 `np.indices` + modulo
  向量化）；大卷積用 `fftconvolve` 取代 `correlate2d`。
- **物理假設寫進註解**：對齊方式、基線百分位、正規化常數、px↔nm 換算等
  都要註明來源與理由（如 SEM 量得、避免退化解）。
- **保持單位一致**：任何涉及 `NORM_MIN/MAX`、`px_nm`、Z-sensitivity 的改動，
  必須同步檢查訓練端與推論端，避免尺度不一致。
- **形態學運算要小心平移不變性**：改動 tip 重建前，先想清楚運算對平移/縮放
  的響應（見 §5）。
- **無頭可測**：數值與繪圖邏輯應能以 `matplotlib.use('Agg')`（或 `xvfb-run`
  跑 Tkinter）在無 GUI 環境驗證；提交前至少 `python -m py_compile` 與一次
  端到端煙霧測試。
- **GUI 執行緒安全**：Tkinter widget 只能在主執行緒操作；背景工作用
  `self.after(...)` 排程回主執行緒（見 `_log_safe`、`_recompute_tip`）。
- **輸出目錄慣例**：沿用 YOLO 風格 `runs/train{N}` / `predict{M}` 自動遞增，
  不要覆寫既有結果。

### 8.3 提交流程 (Definition of Done)
1. `python -m py_compile <changed>.py` 通過。
2. 影響數值/重建邏輯時，跑一次合成資料端到端驗證（如 `compute_tip`、
   `reconstruct`）。
3. **更新 `CLAUDE.md`**（章節 + Changelog）。
4. `git status` 確認無圖片/權重/資料大檔被加入。
5. commit 訊息清楚描述「動機 + 做法」；推送到指定分支。

---

## 9. 變更紀錄 (Changelog)

> 每次更新都在此最上方追加一筆（日期 / 範圍 / 摘要）。

- **2026-07-07 — `blind_deconvolution.py`：自動換算探針視窗半徑（依載入孔深 + R/θ）**
  - **需求**：使用者不想手動換算視窗半徑，希望載入 .000 + 填探針 R/θ 後自動決定。
  - **新增 `auto_tip_half(surface, px_nm, R, θ, sample, cap=60)`**：以物理公式
    `側向延伸 ≈ R·sinθ + D·tanθ` → `半徑px = ceil(側向/px_nm)+1`。其中 `px_nm` 與特徵起伏 D
    兩個樣品相關量從 .000 取得（Scan Size、高度資料；孔洞用 pct95−min、凸起用 pct99.5−pct10），
    R/θ 為探針規格由使用者提供。θ 定義為**半錐角**（全開角需先除以 2）；非對稱取較大一軸角度。
  - **GUI**：「視窗半徑」旁新增「☑ 自動」核取方塊（預設開）；勾選時鎖住手動欄，
    載入影像 / 改 R,θ,px_nm,樣品 時即時重算並填入（trace + self.after）。取消勾選可手動輸入。
  - **CLI**：`--tip-half` 改為可接 `auto`（預設）或整數；auto 時依 `--sample` 起伏 + R/θ 換算。
  - **驗證**：`30nm.003`（256², px_nm=1.17, R=8nm, θ=20.7°, bump）CLI/GUI 皆自動得視窗半徑 5px，
    端到端跑通（certain 49.5%）；GUI 測試確認自動填值、手動欄鎖定/解鎖、切樣品重算皆正確。

- **2026-07-07 — `blind_deconvolution.py`：GUI/CLI 可直接開 Nanoscope `.000` 原始檔回推**
  - **需求**：使用者要在 GUI 直接載入 `.000` 掃描檔做去卷積回推，不必先用 detect.py 轉 `.npy`。
  - **新增 TF-free Nanoscope 載入器**（沿用 detect.py / reconstruct_lines_3d 邏輯）：
    `is_nanoscope_file` / `_parse_nanoscope_header`（Data length 上界防跨通道污染）/
    `_read_height_channel`（Z-scale 校正）/ `_despike`（去孤立尖刺）/ `_flatten_rows`
    （逐行去傾斜，孔洞 pct95、凸起 pct10 基線）/ `load_nanoscope_surface(fp, sample)`。
    **保留原生解析度不 resize**，使影像 px_nm 與探針一致；回傳 (surface_nm, px_nm, info)。
  - **GUI**：① 檔案對話框加入 `.000–.009`；載入 Nanoscope 檔時自動解析、去尖刺、去傾斜、
    **自動帶入 px_nm**；新增「樣品類型 孔洞/凸起」選項（只影響基線方向，erosion 對平移不變，
    不改變還原形狀）。**CLI** 同步：位置參數可為 `.000`，新增 `--sample hole|bump`，
    檔頭 px_nm 覆寫 `--px-nm` 確保探針/影像尺度一致。
  - **驗證**：真實 `std.000` 端到端跑通（解析 ch='Height Sensor'、px_nm=19.53、Z 校正、
    去卷積 s_r≤i、certain 99.9%）；GUI 非阻塞測試確認 .000 分支載入成功並自動帶入 px_nm。
    （註：該 std.000 僅 26/256 行為部分掃描壞檔，載入器正確但幾何不具代表性，建議換掃滿的乾淨檔。）

- **2026-07-07 — `blind_deconvolution.py`：新增步驟式 GUI + 非對稱探針（θx/θy）支援**
  - **GUI（`launch_gui()`，Tkinter）**：把零訓練去卷積流程做成「按步驟操作」介面——
    ① 載入影像(.npy) → ② 設 px_nm → ③ 設探針 → ④ 執行去卷積 → ⑤ 檢視 certainty → ⑥ 儲存。
    右側 2×3 圖：輸入 / 還原(erosion) / certainty map(紅=探針碰不到) / 探針 2D / X 剖面 / Y 剖面。
    去卷積在背景執行緒跑、以 `self.after()` 回主緒更新（沿用 afm_gui 執行緒安全慣例）；
    嵌入採 `FigureCanvasTkAgg`＋`Figure`，不受模組頂 `matplotlib.use('Agg')` 影響（無頭仍可 import）。
    無參數執行即開 GUI（`--gui` 亦可）；`--demo` 與既有 CLI 位置參數行為不變。
  - **非對稱探針 `make_cone_tip_asym(half, px_nm, R, θx, θy)`**：真實探針未必旋轉對稱，
    支援 X/Y 兩軸各自半錐角。以「半錐角隨方位角 φ 橢圓內插」
    `tanθ(φ)=(a·b)/√((b·cosφ)²+(a·sinφ)²)`（a=tanθx, b=tanθy）建構，φ=0 得 θx、φ=90° 得 θy，
    平滑過渡；`θx==θy` 時嚴格退化為對稱 `make_cone_tip`（單元測試 max|diff|≈1.4e-9）。
    GUI 以「☑ 非對稱探針」核取方塊切換單一 θ ↔ θx/θy 兩欄。
  - **核心不動**：`reconstruct_surface`(grey_erosion)/`surface_certainty` 對任意 2D 探針皆成立，
    非對稱只是換一支非旋轉對稱的 structure。煙霧測試：非對稱 tip(θx25/θy10) 下 `s_r ≤ i` 恆成立（不過度去卷積）。
  - CJK 字型 rcParams 移到模組頂（圖中中文正確顯示）。TF-free。

- **2026-06-25 — 新增 `reconstruct_lines_3d.py`：凸起樣品探針去卷積 + 3D 重建（2D/逐線1D）**
  - **需求**：給定探針 R(球冠半徑 nm)/θ(半錐角 °)，對 Nanoscope `.000` 凸起樣品掃描去卷積、
    堆疊成面、輸出 3D 渲染圖。
  - **凸起 vs 凹孔**：凸起頂端被探針準確爬過、只有側壁被撐寬(tip broadening)→ erosion
    可良好還原（凹孔則底部資訊物理遺失，救不回）。erosion 對兩者皆成立，凸起結果更佳。
  - **三種模式（`--mode`）**：
    - `2d`（**預設、推薦**）：`reconstruct_2d()` 用完整 2D 探針 grey_erosion，**X/Y 同時等向還原**。
    - `line`：`reconstruct_per_line()` 逐行 1D erosion——**只修 X(快軸)、Y(慢軸)不動** →
      被等向撐大的圓會還原成**橢圓**（X 收窄、Y 不變）。屬近似，僅供比較。
    - `both`：兩者並排比較。
    - **驗證（圓盤凸起）**：真實 41×41 → 探針撐大 63×63 → 1D 還原 43×63（橢圓！）、
      2D 還原 43×43（圓 ✓）。確認逐線 1D 的非等向缺陷，**正解為 2D**。
  - **作法**：沿用 detect.py 的 Nanoscope 解析/Z 校正（Data length 上界、跨通道保護）；
    `flatten_rows_bumps()` 凸起版基線（低百分位為表面、特徵向上）；`make_cone_tip_2d()`/
    `make_cone_profile_1d()` 生成探針；`auto_tip_half()` 依凸起高度自動估視窗。3D 用 plot_surface
    輸出輸入 vs 還原並排圖。所有模式 `s_r ≤ i` 恆成立=不過度去卷積。TF-free。

- **2026-06-25 — 新增 `blind_deconvolution.py`：通用 certainty 去卷積引擎（零訓練）**
  - **動機**：DL 路線（train14–16）是「為單一探針+單一樣品」訓練，無法當通用產品；
    且診斷顯示對本樣品 DL 幾乎無加值（尖探針失真小、平底資訊物理遺失、且會過度去卷積）。
    通用產品該走 Villarrubia (1997) 形態學：成像 `i=s⊕t`、還原 `s_r=i⊖t`（grey erosion）。
  - **核心（已驗證正確）**：`reconstruct_surface()` 用已知探針對影像做 grey_erosion，
    嚴格滿足 `s ≤ s_r ≤ i`（demo 驗證 True）→ **數學保證不會過度去卷積**（只給確定下界）。
    `surface_certainty()` 標出探針『實際接觸過』的表面點，其餘（如寬孔平底）標為 uncertain
    （demo：孔底 95% 被標不可信）→ 誠實標出物理死角、不腦補。
  - **探針來源**：吃『已知探針』——(A) 廠商 cone（`make_cone_tip(R,θ)`，推薦）或
    (B) Gwyddion `itip_estimate` 跑尖刺校正光柵匯出的 tip.npy。**本檔不自實作 blind 盲估**
    （Villarrubia 迭代易錯，交給 Gwyddion 成熟實作較可靠誠實）。
  - **CLI**：`--demo`（合成驗證）/ `scan.npy --tip tip.npy` / `scan.npy --cone-R 2 --cone-theta 25`。
    在真實 std.002 上用尖 cone 還原=掃描（99.9% certain）—— 誠實回報「尖探針沒失真、無需修正」，
    對比 DL 同案例過度放大到 388nm。給較鈍的真實探針則會修更多並標更多 uncertain。
  - **定位**：此為通用引擎雛形，與 DL pipeline 並存；要打包給所有 AFM 商做影像還原，
    應以此為核心（探針當執行時輸入、certainty 誠實回報），而非為每支探針重訓 DL 模型。

- **2026-06-22 — `Supporting_code_AFM_deep_learning.py`：dilation 改用廠商 cone 探針，根治過度去卷積（input 分布不匹配）**
  - **根因（接續 train15 仍過放大的診斷）**：用絕對深度門檻量測發現，**真正的不匹配在 input 端**——
    訓練 dilated 孔（模型 input）只有 ~88nm@-20nm，但真實掃描 input 孔達 ~253nm@-20nm。模型學到
    「窄 input → 寬 GT」的大放大率，套到本來就寬的真實 input → 外插成 ~388nm（vs SEM 真實 228.8nm）。
    input 太窄是因為 dilation 用的探針（盲重建 46.9nm 鈍探針 / 拋物線）對「孔比探針寬」的樣品
    narrow 過頭（GT 220 → dilated 88）。**先前改 GT 孔大小是改錯層級，故 train15 無改善。**
  - **新增 `make_cone_training_tip(R_nm, theta_deg, max_px, px_nm)`** 與常數
    `USE_CONE_TIP/CONE_R_NM/CONE_THETA_DEG/CONE_MAX_PX`：以廠商規格的「球冠+直線錐壁」尖 cone
    取代鈍探針做 grey_dilation。尖 cone 對寬孔幾乎不 narrow（模擬 dilated 220/202/202 ≈ 真實
    input 253/216/165；舊鈍探針為 132/132/99），訓練 input 終於對得上真實掃描 input。
    `USE_CONE_TIP=True` 時跳過 tip.mat 載入；R/θ 預設 2nm/25°（請依探針 datasheet 調整）。
  - **為何不用盲重建 tip**：本樣品孔寬於探針，盲重建物理上只能 characterize 探針頂端、測不到
    完整錐壁（且唯一可用檔 std.004 資料壞——量測深度 raw 164.7nm>真實、flatten 後又=0，見 afm_gui
    [FIX-12]）。直接用 datasheet cone 最穩健，不需 GUI 重建、不需 tip.mat。
  - **需重新訓練**：改動 dilation 探針，需重跑產生新 `runs/train{N}`；預期預測孔徑收斂到 ~220-260nm。

- **2026-06-22 — `afm_gui.py`：新增 `extend_tip_cone()` 錐壁延伸，修正重建探針外緣攤平變鈍 [FIX-11]**
  - **問題**：重建出的 tip 曲線兩側「趨於平緩（攤平裙邊）」而非錐狀。根因為盲重建
    （`grey_erosion`）只能還原探針**實際接觸孔壁的頂端區段**；接觸區以外 min 運算飽和，
    側壁攤平、探針看起來變鈍。對寬孔（如 std.004 視野內僅 0.7 個特徵、進入率 121.5%）尤其明顯。
  - **新增 `extend_tip_cone(tip, half, px_nm)`**：量測接觸區錐壁斜率（取最陡 30% 為「可靠錐壁」），
    沿該斜率把側壁**直線外推到邊緣（含角落）**，取代攤平裙邊，還原直線錐形側壁。
    在 `compute_tip()` 的 `radial_average` 之後呼叫；ax4(2D)/ax5(cross-section 綠線) 直接反映。
  - **下游效益**：此修正讓重建探針變尖（真正的 cone），存成 tip.mat 餵
    `Supporting_code_AFM_deep_learning.py` 做 dilation 時 narrow 量恢復正常。先前診斷出
    train15 仍過度去卷積（預測孔徑 ~388nm vs SEM 228.8nm），根因正是 dilation 探針太鈍
    （訓練 input 孔 ~88nm 遠窄於真實掃描 input ~253nm@-20nm，模型學到過大放大率）。
    錐壁延伸後的尖探針可改善此 input 分布不匹配，**需以新 tip.mat 重新訓練才生效**。
  - 單元測試：對「球冠+攤平裙邊」鈍探針，修正後外緣斜率持續為負（直線錐壁）、不再攤平，
    探針頂點到邊緣加深（更尖）。
  - **[FIX-12] 物理健全性警告**：`reconstruct()` 新增檢查——孔洞經探針掃描只會變淺，
    量測深度不可能 > 真實深度。若 `meas_depth > true_depth×1.05`（如 `std.004` 量到
    164.7 > 125.8nm）→ 印 ⛔ 警告：高度資料含尖刺/壞值、重建探針無效（會得到中心反而最低
    的顛倒形狀，`extend_tip_cone` 偵測不到向下錐壁而保險跳出），建議換乾淨檔或改用廠商 cone。
  - **適用性結論**：本樣品孔洞「寬於探針」，盲重建物理上只能 characterize 探針頂端、
    無法還原完整錐形側壁；要完整尖探針應直接用 🟠 vendor cone（`make_cone_tip`）存 tip.mat。
    判斷檔案是否可用於重建：量測深度需 < 真實深度（進入率 <100%）。

- **2026-06-18 — `Supporting_code_AFM_deep_learning.py`：圓柱孔尺寸對齊真實樣品，修正橫向過度去卷積（~1.45×）**
  - **根因**：對 5000nm 完整掃描（`std.002`，橫向尺度 39.06 nm/px ≈ 訓練 39.1，**尺度本身相符**）
    推論時，預測孔徑量得約 **8.5 px（~332 nm）**，比 SEM 量得真實開口 **228.8 nm（半徑 2.93 px）大約 1.45×**
    （預測孔洞像素佔比 7.6% vs 輸入 1.0%）。深度則正確（預測 122 nm ≈ 真實 125.8 nm）。
    追根究柢為**訓練圓柱孔的半徑分佈整體偏大**：舊版 `cylinder_hole_randomizer` 半徑 3–24 px
    （70% 3–8px、30% 8–24px），**加權平均半徑 8.65 px（≈676 nm 開口）≈ 真實孔的 3 倍**。
    模型看慣大孔，遇到真實小孔便「腦補」成熟悉的大尺寸 → 系統性橫向過度放大。此為先前多次
    「過度去卷積」紀錄（縮小探針後）仍殘留的最後一塊：探針已修，但孔洞尺寸先驗仍偏大。
  - **修正 `cylinder_hole_randomizer`**：半徑分佈改為以真實孔徑為中心的窄分佈——
    主群 80% `U[2.9, 3.8]`px（227–297 nm，涵蓋真實開口 ±30%）、上緣 20% `U[3.8, 6.0]`px
    （297–469 nm，輕度尺度泛化），**完全移除 8–24px 巨孔群**。新分佈加權平均半徑降為
    3.66 px（≈286 nm），貼近真實 228.8 nm。下限 2.9 px 仍 > 探針填充極限 2.77px@125nm，
    經 grey_dilation 驗證各半徑孔洞膨脹後仍保有可見訊號（minZ≈−125.8 nm），不致退化成
    「全平→生孔」解。`config.txt` 同步更新半徑與深度說明文字。
  - **需重新訓練**：此為訓練資料改動，需重跑 `Supporting_code_AFM_deep_learning.py` 產生新
    `runs/train{N}` 後才生效；舊 train14 權重仍會過度放大。

- **2026-06-16 — `detect.py`：修正 channel 讀取越界（跨 channel 資料污染），新增 `inspect_header.py`**
  - **根因**：`read_nanoscope_channel()` 舊版永遠按 `n_px×n_lines×bpp`（假設跑滿全幅）
    讀取每個 channel 的位元組數，忽略標頭中該 channel 自己宣告的 `Data length` 欄位。
    當掃描中途中斷（如本例 49/256 行即停止）時，每個 channel 實際只有 49 行有效資料，
    但程式仍嘗試讀 256 行的位元組數，導致讀取範圍跨越到下一個 channel 的資料區
    （甚至更後面的 channel），把不相干數值（如 DMTModulus、Adhesion）混入
    `Height Sensor` 影像，造成 std 異常大（6830）、看似雜訊但實為跨通道污染——
    這也是先前 std.001 推論出現「121.5× 過度去卷積」假警報的根本原因
    （輸入經污染後基線校正範圍仍接近 0，模型任何微小輸出都被無限放大解讀）。
  - **修正**：`parse_nanoscope_header()` 新增解析 `Data length` 欄位（`chs[].data_length`）；
    `read_nanoscope_channel()` 讀取時以 `min(n_px×n_lines×bpp, data_length)` 為界，
    不再越界讀到下一個 channel。既有的 `actual_lines` 部分掃描處理邏輯（zoom 縮放至
    128×128）可正確接手，但需注意：若有效行數遠少於宣告行數（如本例僅 19%），
    縮放後的幾何形狀會嚴重失真，建議使用者確認該掃描是否需要重新量測。
  - **新增 `inspect_header.py`**：診斷工具，印出各 channel 的 `Data offset/Bytes per
    pixel/Data length/Z scale` 原始文字與解析結果，並比較「未限制長度」vs
    「以 Data length 為界」兩種讀法的 raw min/max/std，方便日後排查類似的
    部分掃描或 channel 邊界問題。

- **2026-06-16 — `detect.py`：放寬 Nanoscope 副檔名偵測，支援描述性字尾**
  - **問題**：使用者將 `std.000` 手動標記為 `std.000-after flatten`（內容仍是原始 Nanoscope
    二進位格式，僅檔名加註處理階段），但 `is_nanoscope_file()` 舊版用 `^\.\d{3}$`（嚴格匹配
    3 位數字結尾）判斷，`.000-after flatten` 不符合 → 誤判為一般圖片，PIL 開檔失敗
    (`UnidentifiedImageError`)。
  - **修正**：正規表達式改為 `^\.\d{3}`（只要求以 3 位數字開頭，不要求結尾），允許
    `.000`、`.004` 等標準副檔名後接任意描述文字，使用者可自由標記處理階段而不必每次重新命名。

- **2026-06-16 — `Supporting_code_AFM_deep_learning.py` + `afm_gui.py`：最小孔洞半徑修正、訓練物理說明、GUI 分隔符**
  - **`cylinder_hole_randomizer` 最小半徑 1.5 px → 3 px**：舊值 1.5px（59nm）小於探針填充極限半徑
    2.77px（@深度125nm），導致小孔在 dilation 後幾乎完全填平，訓練 Input 趨近全平坦。
    新值 3px（117nm）> 填充極限，確保 dilated 訓練影像有可見孔洞訊號，模型能學到有意義的逆映射。
  - **新增訓練資料物理說明 log**：訓練開始時列印 grey dilation 方向說明（X=dilated 小洞 → y=GT 大洞）、
    有效填充半徑公式與計算範例，方便調整資料集設定時確認物理方向正確。
  - **`afm_gui.py` 重建結果後加分隔符**：每次重建（含失敗）末尾 `_log_safe('='×52)` 使
    log 區中每次執行的結果清楚分隔，不與上一次混在一起。

- **2026-06-15 — `Supporting_code_AFM_deep_learning.py`：縮小訓練探針，修正過度去卷積**
  - **根因**：訓練 tip.mat 被 `load_tip_for_training` 重採樣後為 7×7 px（半寬 3 px = 117 nm），
    而真實孔洞半徑僅 2.93 px（228.8 nm / 2 / 39.1 nm/px）。有效填滿半徑
    `sqrt(2 × R_tip × D) / px_nm = sqrt(2×46.9×125)/39.1 ≈ 2.77 px ≈ 孔洞半徑`，導致
    訓練時孔洞被 100% 填平；模型學到「完全平坦 → 輸出大圓洞」，套到真實 AFM（孔洞可見）時
    過度放大 2–3 倍。
  - **修正 `load_tip_for_training` 呼叫**：加入 `max_tip_px=5` 參數，將訓練探針限制為
    5×5 px（半寬 2 px = 78 nm < 孔洞半徑 114 nm），使小孔在 dilation 後仍有可見訊號。
  - **修正 `TIP_RADIUS_NM`**：73.0 → 46.9 nm（afm_gui 量得真實 ROC），同步更新 fallback 合成探針。
  - **修正 `TIP_TRAIN_SIZE`**：9 px → 5 px，與新 ROC 和 max_tip_px=5 一致。
  - **需重新訓練**：上述三項改動需重跑 `Supporting_code_AFM_deep_learning.py` 才生效。

- **2026-06-15 — `detect.py`：修正 depth_ratio ✓ 判斷邏輯**
  - 舊條件 `0.8 < depth_ratio < 3.0` 對負數有歧義：當 in_min 與 pr_min 皆為負值時，
    ratio = 0.81 表示 |predicted| < |input|（預測孔洞**更淺**），不是更深，舊版卻印 ✓。
  - 修正：`ratio ≥ 1.0`（|預測| > |輸入|）才算正確去卷積方向；`ratio < 1.0` 改印 ⚠，
    並提示常見原因（輸入含大片壞區、Z 超出訓練分布）。同時補充 ratio 物理意義的程式碼註解。

- **2026-06-15 — `detect.py`：新增連續壞區物理裁切（接續去尖刺）**
  - **發現**：`std.000` 去尖刺後仍殘留 +5526nm 的**連續壞區**（中值濾波移不掉，因鄰域
    同樣是壞值），且該區污染 `flatten_rows_holes` 的 percentile(95) 基準，仍觸發 Z 全圖
    縮放、壓扁真實孔洞（深度比 0.63×）。此壞區與部分掃描（208/256 行）有關。
  - **新增 `clip_physical_range()`**：以表面中位數為基準，裁切到物理 Z 窗
    `[median−300nm, median+60nm]`。孔洞樣品表面為高基準、特徵向下，故高側收緊
    （表面之上僅 ±20nm 條紋雜訊）、低側放寬保留深孔；連續壞區被夾到表面附近視為
    平坦表面（無意義但無害）。在 `despike_image` 後、`flatten_rows_holes` 前呼叫。
  - **注意**：此為穩健化處理，部分掃描/壞檔仍建議改用掃滿 256 行的乾淨檔案。

- **2026-06-15 — `detect.py`：新增輸入去尖刺（despike），根治「孔洞預測超大」**
  - **根因**：真實掃描 `std.000` 含非物理尖刺（單張 Z 範圍達 6216nm，min≈-5404nm），
    觸發舊版 `×0.045` 全圖 Z 縮放保護 → 真實 ~150nm 孔洞被壓成 ~7nm（近乎消失），
    模型只看得到被放大的尖刺，輸出巨大假孔洞。尖刺可能與部分掃描（Height Sensor
    只讀到 208/256 行）邊界壞值有關。
  - **新增 `despike_image()`**：3×3 中值濾波估計局部基準，殘差超過
    `max(300nm, 8×穩健標準差)` 的孤立像素以局部中值取代。真實深孔/壞線因鄰域同樣
    偏移、殘差小而被保留；只移除非物理尖刺。`load_nanoscope()` 在 `flatten_rows_holes`
    前呼叫，使 Z 範圍回到正常、不再觸發全圖縮放。
  - **修正圖表中文亂碼**：`visualize_results()` suptitle 改用英文，避免無 CJK 字型
    環境（如使用者 Windows + DejaVu Sans）大量 glyph-missing 警告與方框。

- **2026-06-15 — `detect.py`：修正「孔洞預測超大」視覺誤判 + 橫向尺度診斷 + Z 範圍雙層保護**
  - **視覺化共用色階**：`visualize_results()` 改為左右兩圖使用相同 `vmin/vmax`（取輸入/預測聯集範圍），
    並新增第三格差值圖（`predicted - input`，RdBu_r 色彩），讓孔洞深淺校正量一目了然。
    原本各圖獨立 auto-scale 會使淺孔（input）與深孔（predicted）顯示出相同大小的色塊，
    造成「孔洞變超大」的視覺誤判。
  - **新增深度比警告**：`depth_ratio = pr_min / in_min`；≥ 3.0× 時印出過度去卷積警告及三項常見根因（掃描範圍/Z靈敏度/訓練探針不符）。
  - **橫向尺度診斷**：`load_nanoscope()` 計算 `actual_px_nm = scan_nm / 128` 與訓練基準
    `SURFACE_SCALE_NM = 39.1 nm/px`（5000nm/128px）之比率；偏差 >15% 時印出警告與建議動作
    （調整掃描範圍至 5000nm 或重新訓練）。偏差過大時孔洞在像素空間尺寸偏移，模型過補/欠補。
  - **Z 範圍雙層保護**：原有一層（z_range > 10× 訓練範圍才縮放）保持；新增軟性第二層（部分值
    超出 `[NORM_MIN, NORM_MAX]` 時僅警告，不強制縮放，避免正常訊號被意外抑制）。

- **2026-06-03 — `Supporting_code_AFM_deep_learning.py`：移除星形孔訓練資料、提高圓柱孔佔比**
  - **移除星形孔訓練**：`grey_dilation` 是多對一映射——圓形、方形、星形孔洞膨脹後外觀相似，
    混合訓練使模型對圓形輸入「猜測」尖角存在（角狀歧義）。樣品確認為圓形/平滑孔後，
    移除星形孔資料可消除歧義，預測邊緣更圓滑。`star_hole_randomizer/creator` 函式保留
    供未來非凸樣品使用，但不加入訓練迴圈。
  - **N_CYL=1800、N_TRAP=600**（原各 1200）：圓柱孔佔比由 50% 提升至 75%，強化模型對
    圓形/平滑孔的辨識能力；梯形孔縮減以降低方形角落特徵的影響。
  - **總資料 3600→2400 張**（train 2880→1920 / test 720→480）；dilation 迴圈改為
    梯形/圓柱分開兩段（各自長度不同），移除舊版假設三類資料量相同的 `n_data` 共用邏輯。
  - config.txt 同步更新：星形孔來源行改為「已移除」說明，N 值改用 `N_TRAP/N_CYL` 變數。

- **2026-06-01 — `afm_gui.py`：新增「廠商 cone 當 tip」輸出（保留原重建功能）**
  - 側欄新增「🟠 儲存 Cone 為 tip.mat」按鈕；用 ⑤ 的 R/θ 經 `make_cone_tip()`
    生成完整錐形探針（球冠頂端 + 錐形側壁，頂點=0、旋轉對稱），存成 tip.mat。
  - 新增 `_current_cone_tip()`（沿用重建 half，未重建時用 `auto_params` 推算）
    與 `save_cone()`（含 `tip_source='vendor_cone'`、`cone_R_nm/θ` metadata），
    格式與重建 tip 一致，可直接餵訓練 pipeline 的 `load_and_prepare_tip`。
  - 動機（CV/AFM）：寬孔只能 characterize 探針「實際接觸」的頂端區段，無法
    還原細長探針的完整深度（探針一插到底、側壁未碰孔壁）；若訓練需要一支
    完整深長探針做 dilation，直接採用廠商 cone 規格較合適。重建 tip 與 cone
    並存，使用者自行取捨。

- **2026-06-01 — `afm_gui.py`：移除 AFM 曲線深度縮放前處理、修正中文顯示與按鈕**
  - `compute_tip()` 移除深度縮放（`extra/target` 拉伸）與自動底部對齊前處理；
    改為只做最小表面對齊（`gt_p.max() − avg.max()`，通常≈0），保留量測到的
    真實深度，不再對 AFM feature 做任何形狀改變。
  - `manual_offset` 改為純視覺平移（`avg_display = avg_aligned + offset`），
    cross-section 藍線上下移動方便目視判斷進入率，tip 計算不受影響。
  - 修正 Matplotlib CJK 字型：在 `matplotlib.use('TkAgg')` 後立即設定
    `font.sans-serif` 候選清單（JhengHei / PingFang TC / Noto Sans CJK TC），
    解決圖中中文顯示亂碼問題。
  - 修正「▼ 下 / ▲ 上」按鈕無反應：`tk.Scale.set()` 不觸發 `command` callback，
    改為在 `_nudge_offset()` 與 `_reset_offset()` 結尾明確呼叫 `_on_offset_change()`。

- **2026-05-29 — `afm_gui.py`：AFM 曲線手動調整功能初版**
  - 側欄新增「⑥ 手動調整 AFM 曲線」滑桿 + 上下微調 + 歸零；
    抽出 `compute_tip()` 支援 `manual_offset`；`reconstruct()` 額外回傳
    配準後 avg 供快速重算（免重新偵測特徵）。

- **2026-06-03 — `Supporting_code_AFM_deep_learning.py` + `detect.py`：U-Net 架構、正規化修正、訓練穩定化**
  - **換用 U-Net（skip connection）取代 Autoencoder**：原架構 4 次 MaxPool 將空間
    壓縮至 8×8（bottleneck 僅 2048 值），孔洞位置精度嚴重不足。U-Net 3 層編碼器
    +bottleneck(16×16×256=65536值) + 對稱解碼器，每級有 skip connection 直接傳遞
    精確空間特徵，從根本解決預測邊緣鋸齒與大小不一致問題。每卷積塊加入
    BatchNormalization 穩定訓練。
  - **NORM_MIN: -155 → -175 nm**（training + detect.py 同步）：真實 AFM 掃描輸入
    可達 -160nm，加上 ±20nm 掃描條紋模擬後訓練輸入可達 -167nm；原 -155nm 邊界
    導致輸入超出正規化分布，模型看到分布外數值。新值 -175nm 留足安全邊界；
    背景 0nm 正規化值從 0.597 更新為 0.625。
  - **掃描條紋振幅 8nm → 20nm**：真實 AFM 條紋明顯強於原模擬，提升訓練與真實
    掃描的分布匹配度；max_band_px 3→4、n_events 最大 5→8。
  - **新增 ReduceLROnPlateau + EarlyStopping**：val_loss 停滯 100 epoch lr×0.5；
    停滯 400 epoch 自動停止並還原最佳權重；EPOCHS 2000→3000（通常不跑滿）。

- **2026-06-03 — `Supporting_code_AFM_deep_learning.py`：豐富資料集（大孔 + 星形孔）強化魯棒性**
  - **圓柱孔尺度多樣化**：半徑由固定 1.5–8px 改為小孔（70%, 1.5–8px=59–313nm）
    與大孔（30%, 8–24px=313–938nm）混合，避免模型只學到「小圓點」的尺度偏誤；
    深度變異由 80–130nm 加寬至 60–145nm。
  - **新增星形孔（star_hole）**：`star_hole_randomizer/creator` 生成 4/5/6 芒星，
    邊界 `r_bound(θ)=r_mid+amp·cos(n·(θ−θ0))`，提供**非凸 + 含尖角**幾何，
    強迫模型學習各方向去卷積，提升形狀魯棒性；隨機旋轉 θ0 避免方向偏誤。
  - **訓練資料 2 類 → 3 類**：梯形孔 + 圓柱孔 + 星形孔，各 1200 張，
    總計 3600 張（train 2880 / test 720）；三類各自 8:2 切分避免類別偏斜。
  - 新增 `star_hole_preview.png`、`dilated_star_hole_stack.npy`；config.txt 記錄
    三類來源與尺度/深度範圍。

- **2026-06-01 — `Supporting_code_AFM_deep_learning.py` + `detect.py`：使用真實 tip.mat 訓練、增量資料、修正模型路徑自動偵測**
  - **新增 `load_tip_for_training(mat_path, training_px_nm, max_tip_px=21)`**：載入
    `tip.mat`（優先順序：`tip_estimated_corrected.mat` > `tip_estimated.mat` > `tip.mat`），
    讀取 `px_nm` 元數據（`afm_gui.py` 儲存時附帶），以 `zoom_factor = tip_px_nm / training_px_nm`
    重採樣使物理尺寸與訓練影像一致，裁剪至最大 21px（奇數、頂端對齊）後回傳。
    無 tip.mat 時退回合成拋物線探針（`make_training_tip`）。
  - **訓練探針選擇改為自動**：在 grey_dilation 前依優先順序嘗試 `TIP_MAT_CANDIDATES`，
    載入失敗則自動退回合成探針；`tip_source` 字串記錄至 `config.txt`，便於追蹤。
  - **訓練資料量 800 → 1200 張**（每種孔洞類型）：總資料 1600 → 2400；
    train 1280→1920，test 320→480；`config.txt` 改用動態 `{N}` 避免數值不一致。
  - **`detect.py` 新增 `find_latest_model()`**：掃描 `runs/train{N}/weights/` 找最大編號
    的已訓練模型，自動設為預設 `MODEL_PATH`；原本預設路徑 `img/AFM_MAE_autoencoder.keras`
    與訓練輸出路徑不符，導致無 `-m` 參數時靜默找不到模型——此為導致預測異常的主要 bug 之一。

- **2026-05-29 — 新增本 `CLAUDE.md`**
  - 建立專案邏輯敘述、程式架構、CV/AFM 專業要點、倉庫規範（不提交照片、
    不上傳大檔、每次更新需改本檔）與程式開發能力規範（整合 skill 準則）。
