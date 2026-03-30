import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from datetime import datetime, timedelta
import sys
import traceback

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="CausalImpact 工具（纯Python版）", layout="wide")


# ------------------------------------------------------------
# 1. 数据准备函数（兼容日期索引 & 数值索引）
# ------------------------------------------------------------
def prepare_data(df: pd.DataFrame, date_col: str, metric_col: str, treat_flag: str):
    """
    返回:
    - merged_data: 含 test/control 的数据
    - is_datetime: 索引是否为 datetime
    """
    def _looks_like_yyyymmdd(s):
        try:
            datetime.strptime(str(int(s)).zfill(8), "%Y%m%d")
            return True
        except Exception:
            return False

    df = df.copy()
    ser = df[date_col]

    # 先尝试数值化
    can_be_numeric = False
    try:
        ser_num = pd.to_numeric(ser)
        can_be_numeric = True
    except Exception:
        can_be_numeric = False

    if can_be_numeric:
        if ser_num.dropna().apply(_looks_like_yyyymmdd).all():
            df[date_col] = pd.to_datetime(
                ser_num.astype(int).astype(str).str.zfill(8), format="%Y%m%d"
            )
            is_datetime = True
        else:
            df[date_col] = ser_num
            is_datetime = False
    else:
        try:
            df[date_col] = pd.to_datetime(ser, errors="raise")
            is_datetime = True
        except Exception:
            df[date_col] = pd.to_numeric(ser, errors="coerce")
            is_datetime = False

    treatment = (
        df[df[treat_flag] == 1][[date_col, metric_col]]
        .rename(columns={metric_col: "test"})
    )
    control = (
        df[df[treat_flag] == 0][[date_col, metric_col]]
        .rename(columns={metric_col: "control"})
    )

    merged = pd.merge(treatment, control, on=date_col).sort_values(date_col)
    merged = merged.dropna()
    merged = merged.set_index(date_col)

    return merged, is_datetime


def fit_counterfactual(data: pd.DataFrame, pre_period, post_period):
    """
    用干预前数据训练模型：
    test ~ control
    然后预测 post 期 test 的反事实值
    """
    pre_data = data.loc[pre_period[0]:pre_period[1]].copy()
    post_data = data.loc[post_period[0]:post_period[1]].copy()

    if len(pre_data) < 3:
        raise ValueError("观察期数据太少，至少需要 3 条记录。")

    if len(post_data) < 1:
        raise ValueError("表现期没有数据。")

    X_pre = pre_data[["control"]].values
    y_pre = pre_data["test"].values

    model = LinearRegression()
    model.fit(X_pre, y_pre)

    # 预测 pre 和 post 的反事实
    pre_pred = model.predict(X_pre)
    post_pred = model.predict(post_data[["control"]].values)

    # 误差估计
    residuals = y_pre - pre_pred
    sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    # 置信区间（近似 95%）
    ci_low_pre = pre_pred - 1.96 * sigma
    ci_high_pre = pre_pred + 1.96 * sigma
    ci_low_post = post_pred - 1.96 * sigma
    ci_high_post = post_pred + 1.96 * sigma

    result = pd.DataFrame(index=data.index)
    result["actual"] = data["test"]
    result["control"] = data["control"]
    result["predicted"] = np.nan
    result["ci_low"] = np.nan
    result["ci_high"] = np.nan

    result.loc[pre_data.index, "predicted"] = pre_pred
    result.loc[pre_data.index, "ci_low"] = ci_low_pre
    result.loc[pre_data.index, "ci_high"] = ci_high_pre

    result.loc[post_data.index, "predicted"] = post_pred
    result.loc[post_data.index, "ci_low"] = ci_low_post
    result.loc[post_data.index, "ci_high"] = ci_high_post

    result["effect"] = result["actual"] - result["predicted"]
    result["effect_pct"] = np.where(
        result["predicted"] != 0,
        result["effect"] / result["predicted"] * 100,
        np.nan
    )

    return model, result, pre_data, post_data, sigma


def summarize_effect(result: pd.DataFrame, post_data: pd.DataFrame):
    post_result = result.loc[post_data.index].copy()

    actual_sum = post_result["actual"].sum()
    predicted_sum = post_result["predicted"].sum()
    effect_sum = actual_sum - predicted_sum
    effect_pct = effect_sum / predicted_sum * 100 if predicted_sum != 0 else np.nan

    abs_effects = post_result["effect"].dropna()
    rmse = np.sqrt(mean_squared_error(post_result["actual"], post_result["predicted"]))
    mae = mean_absolute_error(post_result["actual"], post_result["predicted"])

    return {
        "actual_sum": actual_sum,
        "predicted_sum": predicted_sum,
        "effect_sum": effect_sum,
        "effect_pct": effect_pct,
        "rmse": rmse,
        "mae": mae,
        "daily_mean_effect": abs_effects.mean(),
        "daily_median_effect": abs_effects.median(),
    }


def plot_result(result: pd.DataFrame, pre_period, post_period, is_datetime: bool):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    x = result.index

    # 上图：实际 vs 反事实
    axes[0].plot(x, result["actual"], label="Actual", color="black", linewidth=2)
    axes[0].plot(x, result["predicted"], label="Predicted Counterfactual", color="blue", linewidth=2)
    axes[0].fill_between(
        x,
        result["ci_low"].values,
        result["ci_high"].values,
        color="blue",
        alpha=0.2,
        label="95% CI"
    )
    axes[0].axvline(pre_period[1], color="red", linestyle="--", label="Intervention Start")
    axes[0].set_title("Actual vs Counterfactual")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：效果
    axes[1].plot(x, result["effect"], label="Effect = Actual - Predicted", color="green", linewidth=2)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].axvline(pre_period[1], color="red", linestyle="--")
    axes[1].set_title("Estimated Effect")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ 上传数据")
    file = st.file_uploader("支持 .csv / .xlsx", ["csv", "xlsx"])

    date_col = metric_col = flag_col = None
    data = None
    is_datetime = True
    run_btn = False

    if file is not None:
        try:
            if file.name.endswith(".csv"):
                raw = pd.read_csv(file)
            else:
                raw = pd.read_excel(file)
        except Exception as e:
            st.error(f"读取文件失败：{e}")
            st.stop()

        cols = list(raw.columns)

        st.header("2️⃣ 选择列")
        date_default_idx = cols.index("dur") if "dur" in cols else 0
        metric_default_idx = cols.index("dau") if "dau" in cols else (1 if len(cols) > 1 else 0)
        flag_default_idx = cols.index("y") if "y" in cols else (2 if len(cols) > 2 else 0)

        date_col = st.selectbox("日期列 (或数值索引列)", cols, index=date_default_idx)
        metric_col = st.selectbox("指标列", cols, index=metric_default_idx)
        flag_col = st.selectbox("treatment 标志列 (0=control, 1=test)", cols, index=flag_default_idx)

        try:
            data, is_datetime = prepare_data(raw, date_col, metric_col, flag_col)
        except Exception as e:
            st.error(f"数据预处理失败：{e}")
            st.stop()

        st.header("3️⃣ 干预时间段")

        if is_datetime:
            idx_min = data.index.min().date()
            idx_max = data.index.max().date()
            total_days = max((idx_max - idx_min).days, 1)

            pre_end_default = idx_min + timedelta(days=total_days // 2)
            post_start_default = idx_min + timedelta(days=3 * total_days // 4)

            pre_range = st.date_input("观察期 (开始, 结束)", [idx_min, pre_end_default])
            post_range = st.date_input("表现期 (开始, 结束)", [post_start_default, idx_max])

            if isinstance(pre_range, list) and len(pre_range) == 2:
                pre_start, pre_end = pre_range
            else:
                st.error("请选择完整的观察期")
                st.stop()

            if isinstance(post_range, list) and len(post_range) == 2:
                post_start, post_end = post_range
            else:
                st.error("请选择完整的表现期")
                st.stop()

            pre_period = [pd.to_datetime(pre_start), pd.to_datetime(pre_end)]
            post_period = [pd.to_datetime(post_start), pd.to_datetime(post_end)]
        else:
            idx_min = int(data.index.min())
            idx_max = int(data.index.max())

            pre_start = st.number_input("观察期开始", value=idx_min)
            pre_end = st.number_input("观察期结束", value=int((idx_min + idx_max) // 2))
            post_start = st.number_input("表现期开始", value=int((idx_min + 3 * idx_max) // 4))
            post_end = st.number_input("表现期结束", value=idx_max)

            pre_period = [pre_start, pre_end]
            post_period = [post_start, post_end]

        st.markdown("---")
        run_btn = st.button("🚀 运行分析")
    else:
        raw = None
        st.info("请上传数据文件 (.csv 或 .xlsx)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
st.title("📈 CausalImpact 在线分析（纯 Python 版）")

if file is None:
    st.info("请先在左侧上传数据文件。")
    st.stop()

st.subheader("原始数据预览")
st.dataframe(raw, use_container_width=True)

st.subheader("整理后的 test / control")
st.dataframe(data, use_container_width=True)

# ------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------
if run_btn:
    if pre_period[0] >= pre_period[1] or post_period[0] >= post_period[1]:
        st.error("开始值必须 < 结束值")
        st.stop()

    if pre_period[1] >= post_period[0]:
        st.warning("观察期结束与表现期开始有重叠或相邻，建议保证干预点清晰。")

    with st.spinner("模型计算中..."):
        try:
            model, result, pre_data, post_data, sigma = fit_counterfactual(data, pre_period, post_period)
            summary = summarize_effect(result, post_data)
        except Exception as e:
            st.error(f"分析失败：{e}")
            st.code(traceback.format_exc())
            st.stop()

    st.success("分析完成！")

    st.subheader("Summary")
    summary_text = f"""
- 干预后实际总和: {summary["actual_sum"]:.4f}
- 干预后预测总和: {summary["predicted_sum"]:.4f}
- 总体效果: {summary["effect_sum"]:.4f}
- 效果百分比: {summary["effect_pct"]:.2f}%
- RMSE: {summary["rmse"]:.4f}
- MAE: {summary["mae"]:.4f}
- 日均效果: {summary["daily_mean_effect"]:.4f}
- 日中位效果: {summary["daily_median_effect"]:.4f}
"""
    st.text(summary_text)

    st.subheader("Plot")
    fig = plot_result(result, pre_period, post_period, is_datetime)
    st.pyplot(fig)

    st.subheader("结果明细")
    st.dataframe(result, use_container_width=True)
