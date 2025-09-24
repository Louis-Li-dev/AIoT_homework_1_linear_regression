
# HW1 報告 — 簡單線性迴歸 (CRISP-DM)

**姓名：** 李安旭
**學號：** 4111056002
**科系：** 資工四

---

## 一、作業題目

本次作業 HW1 要求以 Python 程式實作一個簡單的線性迴歸 (Simple Linear Regression) 範例，並依據 **CRISP-DM (Cross Industry Standard Process for Data Mining)** 流程進行。程式需提供介面讓使用者修改迴歸模型中參數 (斜率 a、截距 b、雜訊大小 noise、資料點數 n)，並且需能以 **Streamlit 或 Flask** 架構進行網頁化部署。

另外，題目要求除了程式與結果之外，必須包含「提示 (Prompt) 與過程 (Process)」，因此我也將我使用 ChatGPT 的過程與提示一併記錄於本報告。

Prompt 分享連結：[https://chatgpt.com/share/68d3491b-2e6c-8013-aabd-0802fc6a65e8](https://chatgpt.com/share/68d3491b-2e6c-8013-aabd-0802fc6a65e8)

Deplayment 分享連結：[https://aiot-homework-1-linear-regression.streamlit.app/](https://aiot-homework-1-linear-regression.streamlit.app/)

---

## 二、CRISP-DM 流程

### (1) 商業理解 (Business Understanding)

目標是利用簡單線性迴歸建立一個「可互動的教學系統」，幫助使用者理解線性迴歸如何從資料中學習出斜率與截距，並比較「真實值」與「模型估計值」。

### (2) 資料理解 (Data Understanding)

這裡的資料不是現實世界的數據，而是由程式自動產生。資料的生成方式是：

$$
y = a \cdot x + b + \varepsilon
$$

其中：

* $a$：斜率 (可由使用者設定)
* $b$：截距 (可由使用者設定)
* $\varepsilon$：雜訊 (依據 Gaussian 分布產生)
* $n$：資料點數 (可由使用者設定)

這樣的設計可以讓我們控制資料分布，並檢驗模型的學習效果。

### (3) 資料準備 (Data Preparation)

程式會依使用者設定的參數產生 $X, y$ 資料集，並使用 `train_test_split` 將資料切分為訓練集與測試集，以利模型評估。

### (4) 建模 (Modeling)

使用 `scikit-learn` 套件中的 `LinearRegression` 進行模型訓練，並計算預測值。為了更直觀展示結果，系統會繪製「測試資料散點圖」與「迴歸直線」。

### (5) 評估 (Evaluation)

模型評估指標包含：

* RMSE (Root Mean Squared Error)
* R² (決定係數)

同時，系統也會比較「真實斜率/截距」與「模型估計值」，幫助使用者觀察差異。

### (6) 部署 (Deployment)

本作業提供兩種 Web 框架：

* **Streamlit**：提供互動式側邊欄控制項，滑桿/輸入框調整參數，並即時更新結果。
* **Flask**：提供簡單 HTML 表單，讓使用者輸入參數，並回傳結果與圖表。

---

## 三、程式設計過程

1. **資料生成模組**
   使用 dataclass 定義參數 `GenConfig`，並撰寫 `make_linear_data()` 函數，能根據使用者輸入的 a, b, noise, n 等生成數據。

2. **模型訓練模組**
   撰寫 `fit_linear_regression()`，將資料分割為訓練/測試集，並進行線性迴歸擬合，回傳模型與評估指標。

3. **Streamlit 介面**

   * 在側邊欄加入輸入控制項 (斜率、截距、雜訊、資料點數、隨機種子、測試集比例)
   * 即時生成圖表與評估指標
   * 可視化：測試資料點與迴歸直線

4. **Flask 介面**

   * HTML 表單輸入參數
   * 後端運算後回傳指標與圖表 (以 Base64 編碼嵌入網頁)

---

## 四、執行方式

### (1) 安裝環境

```bash
pip install -r requirements.txt
```

### (2) 執行 Streamlit

```bash
python app.py --mode streamlit
```

開啟瀏覽器進入 `http://localhost:8501`

### (3) 執行 Flask

```bash
python app.py --mode flask
```

開啟瀏覽器進入 `http://localhost:7860`

---

## 五、結果展示

### Streamlit

* 側邊欄輸入控制參數
* 中央顯示散點圖與迴歸直線
* 右側顯示真實參數、模型估計值、RMSE 與 R²

### Flask

* 表單輸入參數
* 下方顯示評估結果與圖表

---

## 六、心得與結論

這次作業透過 **CRISP-DM 流程**，讓我更系統性地理解資料探勘專案的流程，即使只是簡單的線性迴歸，也可以映射到「商業理解 → 資料生成 → 模型訓練 → 評估 → 部署」。

同時，利用 **Streamlit** 與 **Flask** 建立 Web 介面，讓模型不再只是程式碼與數字輸出，而是能以互動方式呈現，提升學習效果與使用體驗。

