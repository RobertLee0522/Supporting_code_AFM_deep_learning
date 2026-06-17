# 電子研究紀錄簿

## 標題
`std.001` Z 軸校正異常初步診斷與 `inspect_header.py` 診斷工具開發

## 日期
2026-06-16

## 研究目的 / 背景
延續前一筆紀錄（副檔名偵測修正）之後，使用者改用 `std.001-after
flatten` 重新執行推論，程式可正常讀檔，但輸出出現明顯異常：

```
[Z校正] ch='Height Sensor'  z_key=ZsensSens  z_lsb=7.33427e-11  sens=813.1653  → 0.0000 nm/LSB
基線校正後範圍：[-0.0, 0.0] nm
...
預測輸出範圍（nm）: [-0.21, 0.15]
⚠ 深度比 121.50× 偏大，可能過度去卷積
```

輸入影像基線校正後範圍幾乎為 0（看起來像全平坦表面），但模型卻給出
深度比高達 121.5× 的預測，明顯不合理。`z_lsb=7.33427e-11` 數量級
（10⁻¹¹）與正常值（程式預設 fallback 為 `0.000375`，約 10⁻⁴ 量級）
相差約 7 個次方，懷疑為 Z-scale 解析 regex 錯誤所致。

## 方法 / 過程

### 1. 假設：regex 解析錯誤
檢視 `detect.py` 中 `parse_nanoscope_header()` 對 `@2:Z scale:` 欄位的
正則：

```python
zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
```

為直接驗證此 regex 是否誤判，撰寫獨立診斷工具 `inspect_header.py`：
讀取檔案前 65536 bytes 標頭、依 channel 切分區塊，對每個 channel
印出：
- `@2:Z scale:` **原始**標頭文字（未經 regex 擷取）
- regex 解析出的 `z_key` / `z_lsb`
- 該 channel 的 `Bytes/pixel`、`Data offset`、`Data length`
- 實際讀取該 channel 原始整數資料的 `min/max/std/峰峰值`

### 2. 對使用者環境執行診斷
請使用者於本機執行：

```bash
python inspect_header.py "...\flatten\std.001-after flatten"
```

取得輸出，比對 `Height Sensor` channel 原始標頭文字：

```
@2:Z scale: V [Sens. ZsensSens] (0.0000000000733427 V/LSB) 0.315005 V
```

確認 regex 擷取出的 `z_lsb=0.0000000000733427` 與標頭原始文字**完全
一致** —— 排除 regex 解析錯誤的假設，`z_lsb` 異常小是檔案標頭**原文
如實記載**的數值，非程式 bug。

### 3. 物理量級交叉驗證
以括號內第二個數值（hard value）反推可能的資料寬度：

```
0.315005 V / 7.33427e-11 V/LSB ≈ 4.296 × 10⁹
```

此數值接近 2³² ≈ 4.295×10⁹，提示原本懷疑此 channel 可能採 32-bit
（4 bytes/pixel）編碼，而非一般 16-bit。但診斷輸出顯示
`Bytes/pixel = 2`（16-bit），與此猜測不符，故另需查明 `0.315005 V`
與 32-bit 量級接近是否為巧合，或暗示其他層面的資料異常。

### 4. 擴充診斷工具：原始整數統計
進一步擴充 `inspect_header.py`，於解析標頭欄位的同時，實際讀取
（未限制長度）各 channel 的原始整數資料並印出統計量，以判斷資料本身
是否合理：

```
--- Channel #1: Height Sensor ---
  Bytes/pixel : 2
  Data length : 25088
  讀到元素數   : 65536  dtype=int16
  raw min/max  : -32768 / 18330
  raw std      : 6830.00
```

`Data length: 25088 bytes` 對應 `25088 / 2 = 12544` 個 16-bit 元素，
換算 `12544 / 256(px/line) = 49` 行 —— 但程式實際讀取了 `65536`
個元素（即假設跑滿 256 行），**遠超過該 channel 宣告的有效資料量**。
比對 `raw std = 6830`（明顯偏大、近似雜訊分佈），確立新假設：
**程式讀取範圍越界，混入了不屬於此 channel 的資料**。

## 結果 / 數據
1. 排除「regex 解析錯誤」假設：`z_lsb` 數值與標頭原文一致。
2. 確立新發現：所有 channel（`Height Sensor`、`DMTModulus`、
   `Adhesion`、`Deformation`、`Dissipation`、`Height`）的
   `Data length` 欄位皆顯示 `25088 bytes`，對應 `49` 行，
   遠小於標頭宣告的 `256` 行（`Number of lines: 256`）。
3. `read_nanoscope_channel()` 目前永遠依 `n_px × n_lines × bpp`
   （即 256 行的位元組數）讀取，未參照各 channel 自身宣告的
   `Data length` 欄位作為讀取邊界。

## 結論 / 討論
`std.001-after flatten` 這份掃描檔本身是一次**中途中斷的掃描**
（僅完成 49/256 行，約 19%），這是資料層面的客觀事實，非程式錯誤。
然而 `detect.py` 現有讀取邏輯未尊重 `Data length` 邊界，會將讀取範圍
延伸到下一個 channel 的資料區（甚至更後面），造成「跨 channel 資料
污染」——`Height Sensor` 影像實際混入了 `DMTModulus`/`Adhesion` 等
其他 channel 的數值，導致基線校正後範圍計算錯誤（看似全平坦），
這正是後續觀察到 121.5× 過度去卷積假警報的根本原因。

本次診斷成功將問題範圍從「Z 校正參數錯誤」收斂到「channel 讀取邊界
未受限」，為下一筆紀錄（根因修正）奠定基礎。

## 相關檔案
- `inspect_header.py`（新增，本次診斷主要工具）
- `detect.py`（`parse_nanoscope_header()`、`read_nanoscope_channel()`，
  問題定位但尚未修正，留待下一筆紀錄處理）

## 下一步計畫
- 修正 `read_nanoscope_channel()`，使其讀取邊界以各 channel 宣告的
  `Data length` 為準，避免跨 channel 污染（詳見下一筆紀錄）。
- 確認該掃描檔是否需要重新量測以取得完整 256 行資料。
