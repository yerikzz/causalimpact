# app.py
import streamlit as st
import pandas as pd
try:
    import pandas.core.dtypes.common as _pdt
    if not hasattr(_pdt, "is_datetime_or_timedelta_dtype"):
        def is_datetime_or_timedelta_dtype(arr):
            return _pdt.is_datetime64_any_dtype(arr) or _pdt.is_timedelta64_dtype(arr)
        _pdt.is_datetime_or_timedelta_dtype = is_datetime_or_timedelta_dtype
except Exception:
    # 若无法 patch（例如 pandas 内部接口变动极大），则忽略，让后续导入显示真实错误
    pass

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from datetime import datetime, timedelta
import sys

st.set_page_config(page_title="CausalImpact 工具", layout="wide")

# 尝试导入 causalimpact（如果失败会在页面提示）
causalimpact_available = True
try:
    from causalimpact import CausalImpact
except Exception as e:
    causalimpact_available = False
    causalimpact_import_error = str(e)

# ------------------------------------------------------------
# 1. 数据准备函数 (兼容日期索引 & 数值索引)
# ------------------------------------------------------------
def prepare_cia_data(df: pd.DataFrame,
                     date_col: str,
                     metric_col: str,
                     treat_flag: str):
    """
    返回 (cia, is_datetime)
    cia: 2 列 (test, control) 且以 date_col 作为索引的 DataFrame
    is_datetime: 索引是否为 datetime（True/False）
    逻辑：
    • 如果 date_col 能全部转换为数值   → 先视为数值，再检查是否可能是 yyyymmdd
    • 如果不是纯数字，则尝试直接转 datetime
    """
    def _looks_like_yyyymmdd(s):
        try:
            datetime.strptime(str(int(s)).zfill(8), "%Y%m%d")
            return True
        except Exception:
            return False

    ser = df[date_col]

    # --- ① 先尝试数值化 --------------------------
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

    cia = (
        pd.merge(treatment, control, on=date_col)
        .sort_values(date_col)
        .set_index(date_col)
        .astype(float)
    )

    return cia, is_datetime


# ------------------------------------------------------------
# 2. 运行 CausalImpact
# ------------------------------------------------------------
def run_ci(data: pd.DataFrame, pre, post, season):
    ci = CausalImpact(data, pre, post, model_args={"nseasons": season})
    ci.run()
    return ci

def fig_to_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------
# 3. Sidebar — 上传 & 参数
# ------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ 上传数据")
    file = st.file_uploader("支持 .csv / .xlsx", ["csv", "xlsx"])

    date_col = metric_col = flag_col = None
    cia_data = None
    is_datetime = True

    if not causalimpact_available:
        st.error("警告：应用无法导入 causalimpact。若部署失败，可能是该包依赖 R / rpy2 等系统组件，Streamlit Community Cloud 无法安装。错误信息（简略）:")
        st.text(causalimpact_import_error)
        st.markdown("建议：若需要安装 R/rpy2，请使用 Docker 部署（Render / Cloud Run 等）。")

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

        date_col   = st.selectbox("日期列 (或数值索引列)", cols, index=date_default_idx)
        metric_col = st.selectbox("指标列", cols, index=metric_default_idx)
        flag_col   = st.selectbox("treatment 标志列 (0=control, 1=test)", cols, index=flag_default_idx)

        # 先做一次预处理以便下面动态展示
        try:
            cia_data, is_datetime = prepare_cia_data(raw.copy(), date_col, metric_col, flag_col)
        except Exception as e:
            st.error(f"数据预处理失败，请检查列名与列值。错误：{e}")
            st.stop()

        st.header("3️⃣ 干预时间段")

        if is_datetime:
            idx_min = cia_data.index.min().date()
            idx_max = cia_data.index.max().date()

            total_days = (idx_max - idx_min).days if (idx_max - idx_min).days > 0 else 1
            pre_end_default   = idx_min + timedelta(days=total_days // 2)
            post_start_default = idx_min + timedelta(days=3 * total_days // 4)

            pre_range  = st.date_input("观察期 (开始, 结束)",  [idx_min, pre_end_default])
            post_range = st.date_input("表现期 (开始, 结束)",  [post_start_default, idx_max])

            if isinstance(pre_range, tuple) and len(pre_range) == 2:
                pre_start,  pre_end  = pre_range
            else:
                st.error("请选择完整的观察期开始和结束日期")
                st.stop()

            if isinstance(post_range, tuple) and len(post_range) == 2:
                post_start, post_end = post_range
            else:
                st.error("请选择完整的表现期开始和结束日期")
                st.stop()

            pre_period  = [pd.to_datetime(pre_start),  pd.to_datetime(pre_end)]
            post_period = [pd.to_datetime(post_start), pd.to_datetime(post_end)]

        else:
            idx_min, idx_max = int(cia_data.index.min()), int(cia_data.index.max())
            pre_start  = st.number_input("观察期开始",  value=idx_min)
            pre_end    = st.number_input("观察期结束",  value=int((idx_min + idx_max) // 2))
            post_start = st.number_input("表现期开始",  value=int((idx_min + 3 * idx_max) // 4))
            post_end   = st.number_input("表现期结束",  value=idx_max)

            pre_period  = [pre_start,  pre_end]
            post_period = [post_start, post_end]

        season = st.number_input("季节性 nseasons", 1, 365, value=7)
        st.markdown("---")
        run_btn = st.button("🚀 运行 CausalImpact")
    else:
        raw = None
        run_btn = False
        st.info("请上传数据文件 (.csv 或 .xlsx)")

# ------------------------------------------------------------
# 4. 主页面显示
# ------------------------------------------------------------
st.title("📈 CausalImpact 在线分析")

if file is None:
    st.info("请先在左侧上传数据文件。")
    st.stop()

st.subheader("原始数据预览")
st.dataframe(raw)

st.subheader("整理后的 test / control")
st.dataframe(cia_data)

# ------------------------------------------------------------
# 运行模型
# ------------------------------------------------------------
if run_btn:
    if not causalimpact_available:
        st.error("当前部署环境无法 import causalimpact。请参考侧边栏的错误信息，或改用支持系统依赖的部署（Render / Cloud Run / Docker）。")
        st.stop()

    # 基础合法性检查
    if pre_period[0] >= pre_period[1] or post_period[0] >= post_period[1]:
        st.error("开始值必须 < 结束值")
        st.stop()

    with st.spinner("模型计算中 ..."):
        try:
            ci = run_ci(cia_data, pre_period, post_period, int(season))
        except Exception as e:
            st.error(f"模型运行失败：{e}")
            st.stop()

    st.success("模型完成！")
    st.subheader("Summary")
    buffer = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buffer
        ci.summary()
    finally:
        sys.stdout = old_stdout
    summary_str = buffer.getvalue()
    st.text(summary_str)

    st.subheader("Plot")
    try:
        fig = ci.plot(figsize=(10, 6))
        st.pyplot(fig)
    except Exception as e:
        st.error(f"绘图失败：{e}")
