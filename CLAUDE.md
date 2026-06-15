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

### 3.1 `afm_gui.py`（盲探針重建 GUI）
- `parse_nanoscope()` / `read_channel()`：解析 Nanoscope 二進位標頭，
  讀取高度通道，套用 Z-scale（`raw × z_lsb × z_sensitivity → nm`）。
- `flatten_rows()`：逐列一階多項式去傾斜；孔洞型以 percentile(95) 為基線。
- `make_gt()` / `make_gt_patch()`：依 SEM 幾何向量化生成理想孔洞 GT。
- `reconstruct()`：偵測特徵 → 收集 patch → 平均 → 2D 配準 → 深度對齊 →
  形態學腐蝕 (`compute_tip`)。
- `compute_tip()`：**核心**。表面/基底對齊 GT 後做 Villarrubia 形態學腐蝕
  (`grey_erosion`)，再四向對稱化 + 徑向平均強制旋轉對稱。支援
  `manual_offset` 以 **深度縮放** 手動調整 AFM 曲線探底（見 Changelog）。
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
- 合成探針：`make_training_tip()` 拋物面 `z(r) = -r²/(2R)`，頂點=0。
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
