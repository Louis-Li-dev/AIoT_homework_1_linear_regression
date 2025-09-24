import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------
# Data generation
# ------------------------
@dataclass
class GenConfig:
    a: float = 2.0
    b: float = 0.0
    noise_sigma: float = 1.0
    n: int = 200
    x_min: float = -5.0
    x_max: float = 5.0
    random_state: int = 42

def make_linear_data(cfg: GenConfig):
    rng = np.random.default_rng(cfg.random_state)
    x = rng.uniform(cfg.x_min, cfg.x_max, size=cfg.n)
    eps = rng.normal(0.0, cfg.noise_sigma, size=cfg.n)
    y = cfg.a * x + cfg.b + eps
    return x.reshape(-1, 1), y

def fit_linear_regression(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return lr, X_test, y_test, y_pred, rmse, r2

# ------------------------
# Streamlit UI
# ------------------------
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Linear Regression — CRISP-DM Demo", layout="wide")

    st.title("Linear Regression — CRISP-DM Demo")

    with st.sidebar:
        st.header("Data Generation")
        a = st.number_input("Slope (a)", value=2.0)
        b = st.number_input("Intercept (b)", value=0.0)
        noise_sigma = st.slider("Noise σ", 0.0, 10.0, 1.0, 0.1)
        n = st.slider("Number of points", 20, 5000, 200, 10)
        x_min, x_max = st.slider("x range", -20.0, 20.0, (-5.0, 5.0))
        seed = st.number_input("Random seed", value=42)
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)

    cfg = GenConfig(a=a, b=b, noise_sigma=noise_sigma, n=n, x_min=x_min, x_max=x_max, random_state=int(seed))
    X, y = make_linear_data(cfg)
    lr, X_test, y_test, y_pred, rmse, r2 = fit_linear_regression(X, y, test_size=test_size, random_state=int(seed))

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(X_test.ravel(), y_test, label="Test data")
        order = np.argsort(X_test.ravel())
        ax.plot(X_test.ravel()[order], y_pred[order], color="red", label="Fitted line")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.metric("Ground truth slope (a)", f"{a:.4f}")
        st.metric("Estimated slope", f"{lr.coef_[0]:.4f}")
        st.metric("Ground truth intercept (b)", f"{b:.4f}")
        st.metric("Estimated intercept", f"{lr.intercept_:.4f}")
        st.divider()
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R²", f"{r2:.4f}")

# ------------------------
# Flask UI
# ------------------------
def run_flask():
    from flask import Flask, request, render_template_string
    import base64, io

    app = Flask(__name__)

    PAGE = """
    <h1>Linear Regression — CRISP-DM Demo</h1>
    <form method="POST">
      a: <input type="number" step="0.1" name="a" value="{{a}}"> |
      b: <input type="number" step="0.1" name="b" value="{{b}}"> |
      noise σ: <input type="number" step="0.1" name="noise_sigma" value="{{noise_sigma}}"> |
      n: <input type="number" step="1" name="n" value="{{n}}"><br><br>
      <button type="submit">Run</button>
    </form>
    {% if metrics %}
    <h3>Results</h3>
    <pre>{{metrics}}</pre>
    <img src="data:image/png;base64,{{plot_b64}}">
    {% endif %}
    """

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @app.route("/", methods=["GET", "POST"])
    def index():
        a, b, noise_sigma, n = 2.0, 0.0, 1.0, 200
        if request.method == "POST":
            a = float(request.form.get("a", a))
            b = float(request.form.get("b", b))
            noise_sigma = float(request.form.get("noise_sigma", noise_sigma))
            n = int(request.form.get("n", n))

        cfg = GenConfig(a=a, b=b, noise_sigma=noise_sigma, n=n)
        X, y = make_linear_data(cfg)
        lr, X_test, y_test, y_pred, rmse, r2 = fit_linear_regression(X, y)

        metrics = {
            "Estimated slope": f"{lr.coef_[0]:.4f}",
            "Estimated intercept": f"{lr.intercept_:.4f}",
            "RMSE": f"{rmse:.4f}",
            "R²": f"{r2:.4f}",
        }
        fig, ax = plt.subplots()
        ax.scatter(X_test.ravel(), y_test)
        order = np.argsort(X_test.ravel())
        ax.plot(X_test.ravel()[order], y_pred[order], color="red")
        plot_b64 = fig_to_base64(fig)
        return render_template_string(PAGE, a=a, b=b, noise_sigma=noise_sigma, n=n,
                                      metrics=metrics, plot_b64=plot_b64)

    app.run(host="0.0.0.0", port=7860, debug=True)

# ------------------------
# Main CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["streamlit", "flask"], default="streamlit",
                        help="Choose web framework")
    args = parser.parse_args()

    if args.mode == "streamlit":
        run_streamlit()
    else:
        run_flask()
