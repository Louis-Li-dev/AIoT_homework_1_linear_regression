# Linear Regression — CRISP-DM Demo

This project demonstrates **Linear Regression** following the CRISP-DM process.  
It provides two deployment options for interactive exploration:

- **Streamlit app** — rich UI for experimenting with slope, intercept, noise, and dataset size.
- **Flask app** — lightweight web interface for adjusting parameters and viewing results.

---

## ✨ Features
- Synthetic linear data generation with configurable slope `a`, intercept `b`, noise, and number of points.
- Model training and evaluation with `scikit-learn`:
  - Root Mean Squared Error (RMSE)
  - R² Score
- Interactive visualization of regression results.
- Two deployment frameworks:
  - **Streamlit** — data exploration with sliders and metrics panel.
  - **Flask** — simple form-based UI with matplotlib plot.

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Louis-Li-dev/AIoT_homework_1_linear_regression/tree/main
cd linear-reg-crispdm
pip install -r requirements.txt

