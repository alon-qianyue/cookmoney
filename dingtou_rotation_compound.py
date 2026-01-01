#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多资产轮动（股/债/金）——复利口径回测：

与之前“每年20万、当年投入年末收益率”的口径不同，这里是：
- 启动资金 initial_capital（默认200000）
- 资金在组合内复利滚动
- 可选：每年年初追加 annual_contrib（默认0）

轮动信号/执行：
- 使用 cookmoney/dingtou_rotation.py 的信号逻辑（周频、信号滞后1周、可选移动止损）
- 默认参数取你已接受的版本：MA=13周，动量=52周，abs_filter=ma_only，移动止损=10%，费率=0.10%，现金年化=2%

输出：逐年净值与年度收益率（复利口径）。
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_rotation_module():
    """
    以文件路径方式加载 dingtou_rotation.py（无需把 cookmoney 变成 Python package）。
    """
    path = Path(__file__).resolve().parent / "dingtou_rotation.py"
    spec = importlib.util.spec_from_file_location("dingtou_rotation", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass 在 Python3.8 里会通过 sys.modules 查模块命名空间，这里必须提前注册
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@dataclass(frozen=True)
class PortfolioYear:
    year: int
    start_value: float
    end_value: float
    simple_return: float


def simulate_portfolio(
    weekly_prices: pd.DataFrame,
    signal: pd.Series,
    start_year: int,
    end_year: int,
    initial_capital: float,
    annual_contrib: float,
    cash_rate_annual: float,
    fee_rate: float,
    signal_lag_weeks: int,
    trailing_stop_dd: Optional[float],
) -> List[PortfolioYear]:
    wp = weekly_prices.sort_index()
    widx = wp.index[(wp.index.year >= start_year) & (wp.index.year <= end_year)]
    if widx.empty:
        return []

    cash_r_w = (1.0 + float(cash_rate_annual)) ** (1.0 / 52.0) - 1.0

    cash = float(initial_capital)
    holding: Optional[str] = None
    shares = 0.0
    peak_price: Optional[float] = None

    def value_at(date: pd.Timestamp) -> float:
        v = cash
        if holding is not None and shares > 0:
            px = wp.loc[date, holding]
            if pd.notna(px) and px > 0:
                v += shares * float(px)
        return float(v)

    years: List[PortfolioYear] = []
    cur_year = int(widx[0].year)
    year_start_value = value_at(widx[0])
    prev_d = widx[0]

    for d in widx:
        # 年切换：记录上一年，年初追加
        if int(d.year) != cur_year:
            year_end_value = value_at(prev_d)
            years.append(
                PortfolioYear(
                    year=cur_year,
                    start_value=float(year_start_value),
                    end_value=float(year_end_value),
                    simple_return=float((year_end_value - year_start_value) / year_start_value if year_start_value > 0 else 0.0),
                )
            )
            cur_year = int(d.year)
            if annual_contrib > 0:
                cash += float(annual_contrib)
            year_start_value = value_at(d)

        # 现金计息
        if cash > 0:
            cash *= 1.0 + cash_r_w

        # 目标（信号滞后）
        sig_t = d - pd.Timedelta(weeks=signal_lag_weeks)
        target = signal.get(sig_t, None)

        # 价格检查
        if holding is not None:
            px_h = wp.loc[d, holding]
            if pd.isna(px_h) or px_h <= 0:
                prev_d = d
                continue
            px_h = float(px_h)
        else:
            px_h = None

        if target is not None:
            px_t = wp.loc[d, target]
            if pd.isna(px_t) or px_t <= 0:
                prev_d = d
                continue
            px_t = float(px_t)
        else:
            px_t = None

        # 移动止损（基于当前周收盘）
        if holding is not None and shares > 0 and px_h is not None:
            peak_price = px_h if peak_price is None else max(peak_price, px_h)
            if trailing_stop_dd is not None and peak_price is not None:
                if px_h <= peak_price * (1.0 - float(trailing_stop_dd)):
                    sell_amt = shares * px_h
                    cash += sell_amt * (1.0 - fee_rate)
                    shares = 0.0
                    holding = None
                    peak_price = None
                    prev_d = d
                    continue

        # 不变：持有
        if target == holding:
            prev_d = d
            continue

        # 卖出
        if holding is not None and shares > 0 and px_h is not None:
            sell_amt = shares * px_h
            cash += sell_amt * (1.0 - fee_rate)
            shares = 0.0
            holding = None
            peak_price = None

        # 买入
        if target is not None and cash > 0 and px_t is not None:
            buy_amt = cash * (1.0 - fee_rate)
            shares = buy_amt / px_t
            cash = 0.0
            holding = target
            peak_price = px_t

        prev_d = d

    # 收尾最后一年
    year_end_value = value_at(widx[-1])
    years.append(
        PortfolioYear(
            year=cur_year,
            start_value=float(year_start_value),
            end_value=float(year_end_value),
            simple_return=float((year_end_value - year_start_value) / year_start_value if year_start_value > 0 else 0.0),
        )
    )
    return years


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--universe", type=str, default="hs300_etf,treasury10y_etf,gold_etf")
    p.add_argument("--mom-weeks", type=int, default=52)
    p.add_argument("--ma-weeks", type=int, default=13)
    p.add_argument("--abs-filter", type=str, default="ma_only")
    p.add_argument("--signal-lag-weeks", type=int, default=1)
    p.add_argument("--trailing-stop-dd", type=float, default=0.10)
    p.add_argument("--cash-rate-annual", type=float, default=0.02)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--initial-capital", type=float, default=200000.0)
    p.add_argument("--annual-contrib", type=float, default=0.0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    rot = load_rotation_module()
    ETF_PRESETS = rot.ETF_PRESETS
    build_weekly_prices = rot.build_weekly_prices
    compute_signal = rot.compute_signal

    keys = [x.strip() for x in args.universe.split(",") if x.strip()]
    for k in keys:
        if k not in ETF_PRESETS:
            raise ValueError(f"不支持的 universe key: {k}，可选：{', '.join(ETF_PRESETS.keys())}")
    universe = {k: ETF_PRESETS[k] for k in keys}

    beg = f"{args.start_year}0101"
    end = f"{args.end_year}1231"
    weekly_prices = build_weekly_prices(universe=universe, beg=beg, end=end)
    signal = compute_signal(
        weekly_prices=weekly_prices,
        mom_weeks=args.mom_weeks,
        ma_weeks=args.ma_weeks,
        abs_filter=args.abs_filter,
    )
    dd = None if args.trailing_stop_dd < 0 else float(args.trailing_stop_dd)

    years = simulate_portfolio(
        weekly_prices=weekly_prices,
        signal=signal,
        start_year=args.start_year,
        end_year=args.end_year,
        initial_capital=args.initial_capital,
        annual_contrib=args.annual_contrib,
        cash_rate_annual=args.cash_rate_annual,
        fee_rate=args.fee_rate,
        signal_lag_weeks=args.signal_lag_weeks,
        trailing_stop_dd=dd,
    )

    df = pd.DataFrame(
        [
            {
                "年份": y.year,
                "年初净值(元)": round(y.start_value, 2),
                "年末净值(元)": round(y.end_value, 2),
                "年度收益率": f"{y.simple_return*100:.2f}%",
            }
            for y in years
        ]
    )
    rets = np.array([y.simple_return for y in years], dtype=float) if years else np.array([])
    summary = "；".join(
        [
            f"样本年数={len(years)}",
            f"正收益年数={int((rets>0).sum()) if rets.size else 0}",
            f"亏损年数={int((rets<0).sum()) if rets.size else 0}",
            f"最差收益率={rets.min()*100:.2f}%" if rets.size else "最差收益率=-",
            f"平均收益率={rets.mean()*100:.2f}%" if rets.size else "平均收益率=-",
            f"期末净值={years[-1].end_value:.2f}元" if years else "期末净值=-",
        ]
    )

    universe_line = "、".join([universe[k][1] for k in keys])
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = "\n".join(
        [
            "### 多资产轮动（股/债/金）——复利口径（启动资金+可选年度追加）",
            "",
            f"**ETF池**：{universe_line}",
            f"**参数**：abs_filter={args.abs_filter}；MA={args.ma_weeks}周；动量={args.mom_weeks}周；信号滞后={args.signal_lag_weeks}周；止损回撤={dd if dd is not None else '关闭'}；费率={args.fee_rate:.2%}",
            f"**稳定理财/现金年化**：{args.cash_rate_annual:.2%}",
            f"**启动资金**：{args.initial_capital:.0f} 元；**年度追加**：{args.annual_contrib:.0f} 元",
            f"**摘要**：{summary}",
            f"**生成时间**：{now_str}",
            "",
            df.to_markdown(index=False),
            "",
        ]
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


