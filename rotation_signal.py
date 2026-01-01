#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮动“执行信号”脚本（给人工照做用）。

策略基于你已接受的股/债/金轮动版本（见 dingtou_rotation_stock_bond_gold_v2.md 的参数）：
- ETF池：510300（沪深300）、511260（十年国债）、518880（黄金）
- 周频：W-FRI last（每周最后一个交易日收盘价）
- 信号滞后：1周（本周执行用上周信号）
- 绝对过滤：ma_only（只要求价格>MA）
- MA窗口：13周
- 动量窗口：52周（相对动量排名用它）
- 移动止损：10%（持仓价格从峰值回撤>=10%则切到“空仓/现金”直到下周再看信号）
- 现金年化：2%（仅用于回测/持有现金的计息假设；对“执行信号”只提示持有现金）

输出：
- 综合指标 risk_score（0~100）：0=现金/不持仓，100=持有风险资产
- 本周目标持仓 target_asset（或 CASH）
- 辅助表：每个资产的 52周动量、是否在MA上方、当前回撤
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

UNIVERSE: Dict[str, Tuple[str, str]] = {
    "hs300_etf": ("1.510300", "沪深300ETF（510300）"),
    "treasury10y_etf": ("1.511260", "十年国债ETF（511260）"),
    "gold_etf": ("1.518880", "黄金ETF（518880）"),
}

# 固化参数（与 v2 结果一致）
MOM_WEEKS = 52
MA_WEEKS = 13
SIGNAL_LAG_WEEKS = 1
TRAILING_STOP_DD = 0.10


@dataclass(frozen=True)
class Signal:
    asof_week: pd.Timestamp
    exec_week: pd.Timestamp
    target_key: Optional[str]  # None代表现金
    risk_score: int  # 0/100
    reason: str


def fetch_daily_close(secid: str, beg: str, end: str, timeout: int = 20) -> pd.Series:
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": beg,
        "end": end,
    }
    r = requests.get(EASTMONEY_KLINE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"东方财富接口无数据：{json.dumps(payload, ensure_ascii=False)[:500]}")
    rows = []
    for k in klines:
        parts = k.split(",")
        if len(parts) < 3:
            continue
        rows.append((pd.to_datetime(parts[0]), float(parts[2])))
    df = pd.DataFrame(rows, columns=["date", "close"]).dropna()
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df.set_index("date")["close"].astype(float)


def to_weekly_last(daily_close: pd.Series) -> pd.Series:
    return daily_close.sort_index().resample("W-FRI").last().dropna()


def build_weekly_prices(start_year: int, end_year: int) -> pd.DataFrame:
    beg = f"{start_year}0101"
    end = f"{end_year}1231"
    series = {}
    for k, (secid, _name) in UNIVERSE.items():
        series[k] = to_weekly_last(fetch_daily_close(secid=secid, beg=beg, end=end))
    return pd.DataFrame(series).sort_index()


def compute_signal(weekly_prices: pd.DataFrame) -> Tuple[Signal, pd.DataFrame]:
    """
    用“上一周”的信号决定“本周”的目标持仓。
    """
    wp = weekly_prices.dropna(how="all").sort_index()
    if wp.shape[0] < max(MOM_WEEKS, MA_WEEKS) + 3:
        raise RuntimeError("周数据不足，无法计算信号")

    mom = wp / wp.shift(MOM_WEEKS) - 1.0
    ma = wp.rolling(MA_WEEKS, min_periods=MA_WEEKS).mean()

    exec_week = wp.index[-1]
    asof_week = exec_week - pd.Timedelta(weeks=SIGNAL_LAG_WEEKS)
    if asof_week not in wp.index:
        # 找到最接近的上一周（对齐周频）
        asof_week = wp.index[-(1 + SIGNAL_LAG_WEEKS)]

    row_px = wp.loc[asof_week]
    row_ma = ma.loc[asof_week]
    row_mom = mom.loc[asof_week]

    # 绝对过滤：ma_only
    ok = (row_px > row_ma).fillna(False)
    candidates = row_mom[ok].dropna()
    if candidates.empty:
        chosen = None
        reason = "无资产满足：价格>MA(13周)"
    else:
        chosen = str(candidates.idxmax())
        reason = "满足价格>MA(13周)，且52周动量最高"

    # 移动止损检查（用“执行周”价格相对持仓峰值，属于盘中/盘后风控；这里用周收盘近似）
    # 简化：如果 chosen 不是 None，则用该资产最近一段时间最高周收盘作为 peak，算当前回撤
    dd_now = np.nan
    if chosen is not None:
        s = wp[chosen].dropna()
        peak = float(s.loc[:asof_week].max())
        cur = float(s.loc[asof_week])
        dd_now = (peak - cur) / peak if peak > 0 else np.nan
        if not np.isnan(dd_now) and dd_now >= TRAILING_STOP_DD:
            chosen = None
            reason = f"触发移动止损：回撤{dd_now:.0%}>= {TRAILING_STOP_DD:.0%}，本周持有现金"

    risk_score = 0 if chosen is None else 100
    sig = Signal(asof_week=asof_week, exec_week=exec_week, target_key=chosen, risk_score=risk_score, reason=reason)

    # 输出辅助表（以asof_week为基准）
    info = pd.DataFrame(
        {
            "价格": row_px,
            "MA(13周)": row_ma,
            "在MA上方": ok,
            "动量(52周)": row_mom,
        }
    )
    # 计算各资产asof周相对过去峰值回撤
    dds = {}
    for k in wp.columns:
        s = wp[k].dropna()
        peak = float(s.loc[:asof_week].max())
        cur = float(s.loc[asof_week]) if asof_week in s.index else np.nan
        dds[k] = (peak - cur) / peak if peak and not np.isnan(cur) else np.nan
    info["回撤(相对历史峰值)"] = pd.Series(dds)
    info = info.sort_values(["在MA上方", "动量(52周)"], ascending=[False, False])
    return sig, info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2014, help="多取几年用于计算52周动量/均线")
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    weekly_prices = build_weekly_prices(start_year=args.start_year, end_year=args.end_year)
    sig, info = compute_signal(weekly_prices)

    target = "CASH（稳定理财/货基/短债）" if sig.target_key is None else UNIVERSE[sig.target_key][1]
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = "\n".join(
        [
            "### 轮动执行信号（股/债/金）",
            "",
            f"**生成时间**：{now_str}",
            f"**信号基准周（as-of）**：{sig.asof_week.date()}（用于决策）",
            f"**执行周（exec）**：{sig.exec_week.date()}（本周执行）",
            "",
            f"**综合指标 risk_score**：{sig.risk_score}/100",
            f"**本周目标持仓**：{target}",
            f"**原因**：{sig.reason}",
            "",
            "#### 资产状态（以as-of周计算）",
            "",
            info.assign(
                **{
                    "价格": info["价格"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-"),
                    "MA(13周)": info["MA(13周)"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-"),
                    "动量(52周)": info["动量(52周)"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-"),
                    "回撤(相对历史峰值)": info["回撤(相对历史峰值)"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-"),
                }
            ).to_markdown(),
            "",
            "#### 执行提示（人工下单）",
            "",
            "- 若 **risk_score=0**：本周不持有股票/黄金/国债ETF，资金放在稳定理财/货基/短债",
            "- 若 **risk_score=100**：本周只持有“本周目标持仓”对应的那只ETF（其余卖出）",
            "",
        ]
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


