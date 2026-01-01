#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300ETF（510300）风险开关（Risk-On/Risk-Off）模型：
- 目的：尽量避开大幅下跌阶段；Risk-Off 期间资金进入“稳定理财”
- 稳定理财定义：年化≈通胀（默认 2%），即“跑赢通胀就行”

执行口径：
- 周频（W-FRI last），信号滞后 1 周执行（避免上帝视角）
- 每年投入 200,000 元，按周等额入金
- 费率：买/卖各收一次（默认 0.10%）

默认信号（可解释、可观测）：
- Risk-On（允许持仓）需同时满足：
  1) 收盘 > MA(40周)
  2) 12个月动量(52周) > 0
- 风险保护（意外应对）：
  - 持仓后若从峰值回撤 >= dd（默认15%）→ 强制清仓到稳定理财
  - 若Risk-Off则保持现金计息，直到重新满足Risk-On再入场

输出：每个自然年“当年投入”的年末收益率表 + 摘要。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


@dataclass(frozen=True)
class YearResult:
    year: int
    weeks: int
    invested: float
    end_value: float
    simple_return: float


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


def summarize(results: List[YearResult]) -> Dict[str, str]:
    if not results:
        return {"样本年数": "0"}
    rets = np.array([r.simple_return for r in results], dtype=float)
    years = [r.year for r in results]
    worst_i = int(np.argmin(rets))
    return {
        "样本年数": str(int(rets.size)),
        "正收益年数": str(int((rets > 0).sum())),
        "亏损年数": str(int((rets < 0).sum())),
        "最差年份": str(years[worst_i]),
        "最差收益率": f"{rets[worst_i]*100:.2f}%",
        "平均收益率": f"{rets.mean()*100:.2f}%",
        "中位数收益率": f"{np.median(rets)*100:.2f}%",
    }


def simulate_year(
    year: int,
    weekly: pd.Series,
    ma: pd.Series,
    mom: pd.Series,
    dd: float,
    yearly_cash: float,
    cash_rate_annual: float,
    fee_rate: float,
    signal_lag_weeks: int,
) -> Optional[YearResult]:
    w = weekly[weekly.index.year == year]
    if w.empty or int(w.shape[0]) < 45:
        return None

    weeks = int(w.shape[0])
    deposit = yearly_cash / weeks
    cash = 0.0
    shares = 0.0
    peak: Optional[float] = None

    cash_r_w = (1.0 + float(cash_rate_annual)) ** (1.0 / 52.0) - 1.0

    for d, px in w.items():
        if cash > 0:
            cash *= 1.0 + cash_r_w
        cash += deposit
        px = float(px)

        sig_t = d - pd.Timedelta(weeks=signal_lag_weeks)
        ma_t = ma.get(sig_t)
        mom_t = mom.get(sig_t)
        risk_on = (
            ma_t is not None
            and mom_t is not None
            and not np.isnan(ma_t)
            and not np.isnan(mom_t)
            and px > float(ma_t)
            and float(mom_t) > 0.0
        )

        if shares > 0:
            peak = px if peak is None else max(peak, px)
            dd_hit = peak is not None and px <= peak * (1.0 - dd)
            if (not risk_on) or dd_hit:
                sell_amt = shares * px
                cash += sell_amt * (1.0 - fee_rate)
                shares = 0.0
                peak = None

        if shares == 0 and risk_on and cash > 0:
            buy_amt = cash * (1.0 - fee_rate)
            shares = buy_amt / px
            cash = 0.0
            peak = px
        elif shares > 0 and risk_on and cash > 0:
            buy_amt = cash * (1.0 - fee_rate)
            shares += buy_amt / px
            cash = 0.0

    if cash > 0:
        cash *= 1.0 + cash_r_w
    end_value = float(cash + shares * float(w.iloc[-1]))
    return YearResult(
        year=int(year),
        weeks=weeks,
        invested=float(yearly_cash),
        end_value=end_value,
        simple_return=float((end_value - yearly_cash) / yearly_cash),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--yearly-cash", type=float, default=200000.0)
    p.add_argument("--cash-rate-annual", type=float, default=0.02, help="稳定理财年化（通胀代理）")
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--signal-lag-weeks", type=int, default=1)
    p.add_argument("--ma-weeks", type=int, default=40)
    p.add_argument("--mom-weeks", type=int, default=52)
    p.add_argument("--dd", type=float, default=0.15, help="回撤止损阈值（如0.15）")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    secid = "1.510300"
    beg = f"{args.start_year}0101"
    end = f"{args.end_year}1231"
    weekly = to_weekly_last(fetch_daily_close(secid=secid, beg=beg, end=end))

    ma = weekly.rolling(args.ma_weeks, min_periods=args.ma_weeks).mean()
    mom = weekly / weekly.shift(args.mom_weeks) - 1.0

    results: List[YearResult] = []
    for y in range(args.start_year, args.end_year + 1):
        r = simulate_year(
            year=y,
            weekly=weekly,
            ma=ma,
            mom=mom,
            dd=float(args.dd),
            yearly_cash=args.yearly_cash,
            cash_rate_annual=args.cash_rate_annual,
            fee_rate=args.fee_rate,
            signal_lag_weeks=args.signal_lag_weeks,
        )
        if r is not None:
            results.append(r)

    summ = summarize(results)
    summ_line = "；".join([f"{k}={v}" for k, v in summ.items()])
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(
        [
            {
                "年份": r.year,
                "周数": r.weeks,
                "投入(元)": round(r.invested, 2),
                "年末市值(元)": round(r.end_value, 2),
                "当年收益率": f"{r.simple_return*100:.2f}%",
            }
            for r in results
        ]
    )
    content = "\n".join(
        [
            "### 沪深300ETF（510300）——风险开关（进股市/空仓到稳定理财）",
            "",
            f"**Risk-On**：收盘>MA({args.ma_weeks}周) 且 动量({args.mom_weeks}周)>0（信号滞后{args.signal_lag_weeks}周执行）",
            f"**Risk-Off**：否则进入稳定理财（年化={args.cash_rate_annual:.2%}，通胀代理）",
            f"**意外应对**：持仓从峰值回撤>= {args.dd:.0%} → 强制清仓到稳定理财",
            f"**费率**：{args.fee_rate:.2%}（买/卖各一次）",
            f"**摘要**：{summ_line}",
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


