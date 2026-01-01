import datetime as dt
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# Streamlit 运行时 sys.path 可能不包含项目根目录，导致 `import cookmoney` 失败。
# 这里显式把项目根目录加入 sys.path，确保可导入。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cookmoney.web.engine import (
    ETF_PRESETS,
    RotationParams,
    backtest_rotation_compound,
    backtest_rotation_yearly_inflow,
    build_weekly_prices,
    latest_rotation_signal,
)


st.set_page_config(page_title="CookMoney 回测与信号", layout="wide")


st.title("CookMoney：回测系统 + 下一周信号（可量化、可执行）")

st.markdown(
    "\n".join(
        [
            "你可以：",
            "- 设置**期初金额**、选择**策略**、选择**时间段**，生成对应收益率表",
            "- 查看基于最新数据的**未来一周买/卖信号**（含综合指标 `risk_score`）",
        ]
    )
)


with st.sidebar:
    st.header("参数")
    start_year = st.number_input("开始年份", min_value=2005, max_value=2035, value=2016, step=1)
    end_year = st.number_input("结束年份", min_value=2005, max_value=2035, value=2025, step=1)

    st.divider()
    st.subheader("策略：股/债/金轮动（推荐）")
    initial_capital = st.number_input("期初金额（元）", min_value=0.0, value=200000.0, step=10000.0, format="%.0f")
    annual_contrib = st.number_input("年度追加（元，可选）", min_value=0.0, value=0.0, step=10000.0, format="%.0f")

    st.caption("说明：当前Web版先提供轮动策略的“信号面板”。复利/追加的完整回测表会在下一步加到页面里。")

    st.divider()
    st.subheader("轮动参数（可改）")
    mom_weeks = st.selectbox("动量窗口（周）", [13, 26, 52], index=2)
    ma_weeks = st.selectbox("均线窗口（周）", [13, 20, 40, 52], index=0)
    abs_filter = st.selectbox("绝对过滤", ["ma_only", "mom_only", "ma_and_mom", "top_mom_pos"], index=0)
    trailing_stop_dd = st.selectbox("移动止损回撤", ["关闭", "8%", "10%", "12%", "15%", "20%"], index=2)
    signal_lag_weeks = st.selectbox("信号滞后（周）", [0, 1, 2], index=1)
    cash_rate_annual = st.number_input("稳定理财/现金年化（通胀代理）", min_value=0.0, max_value=0.20, value=0.02, step=0.005, format="%.3f")
    fee_rate = st.number_input("交易费率（单边）", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")

    st.divider()
    st.subheader("资产池")
    default_keys = ["hs300_etf", "treasury10y_etf", "gold_etf"]
    universe_keys = st.multiselect(
        "选择ETF（同币种）",
        options=list(ETF_PRESETS.keys()),
        default=default_keys,
    )


if start_year > end_year:
    st.error("开始年份不能大于结束年份")
    st.stop()

if not universe_keys:
    st.error("请至少选择一个ETF")
    st.stop()


@st.cache_data(ttl=3600)
def load_weekly_prices(_keys: List[str], _start_year: int, _end_year: int) -> pd.DataFrame:
    universe = {k: ETF_PRESETS[k] for k in _keys}
    # 多取一年缓冲以计算MA/动量
    return build_weekly_prices(universe=universe, start_year=_start_year - 2, end_year=_end_year)


weekly_prices = load_weekly_prices(universe_keys, int(start_year), int(end_year))

dd = None
if trailing_stop_dd != "关闭":
    dd = float(trailing_stop_dd.replace("%", "")) / 100.0

p = RotationParams(
    mom_weeks=int(mom_weeks),
    ma_weeks=int(ma_weeks),
    abs_filter=str(abs_filter),
    signal_lag_weeks=int(signal_lag_weeks),
    trailing_stop_dd=dd,
    fee_rate=float(fee_rate),
    cash_rate_annual=float(cash_rate_annual),
)


st.header("未来一周信号")
asof_week, exec_week, target_key, risk_score, reason, info = latest_rotation_signal(weekly_prices, p)

col1, col2, col3 = st.columns(3)
col1.metric("risk_score", f"{risk_score}/100")
col2.metric("信号基准周（as-of）", str(asof_week.date()))
col3.metric("执行周（exec）", str(exec_week.date()))

target_display = "CASH（稳定理财/货基/短债）" if target_key is None else ETF_PRESETS[target_key][1]
st.write(f"**本周目标持仓**：{target_display}")
st.write(f"**原因**：{reason}")

st.dataframe(info, use_container_width=True)

st.caption(
    "执行建议：每周最后一个交易日收盘后/周末更新一次看板；risk_score=100 则只持有目标ETF，risk_score=0 则持有稳定理财。"
)

st.divider()
st.header("历史数据（逐年收益率）")

mode = st.selectbox("回测口径", ["每年投入20万（当年投入年末收益率）", "期初资金复利（可选年度追加）"], index=0)
yearly_cash = st.number_input("每年投入金额（元）", min_value=0.0, value=200000.0, step=10000.0, format="%.0f")

with st.spinner("计算中（需要拉取历史行情）..."):
    if mode.startswith("每年投入"):
        df = backtest_rotation_yearly_inflow(
            weekly_prices=weekly_prices,
            p=p,
            start_year=int(start_year),
            end_year=int(end_year),
            yearly_cash=float(yearly_cash),
        )
        df_show = df.copy()
        if "当年收益率" in df_show.columns:
            df_show["当年收益率"] = (df_show["当年收益率"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(df_show, use_container_width=True)
    else:
        df = backtest_rotation_compound(
            weekly_prices=weekly_prices,
            p=p,
            start_year=int(start_year),
            end_year=int(end_year),
            initial_capital=float(initial_capital),
            annual_contrib=float(annual_contrib),
        )
        df_show = df.copy()
        if "年度收益率" in df_show.columns:
            df_show["年度收益率"] = (df_show["年度收益率"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(df_show, use_container_width=True)

st.download_button(
    "下载CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="backtest.csv",
    mime="text/csv",
)


