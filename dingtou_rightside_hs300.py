#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300ETF（510300）：“右侧入场 + 阶段高点用回撤止盈卖出”的可执行回测（按周）。

核心口径：
- 每年投入 200,000 元，按周等额入金（当年周数可能 50~53）
- 周频价格：W-FRI 周最后一个交易日收盘价（Eastmoney 日线重采样 last）
- 信号滞后 1 周：用上周数据生成信号，本周收盘执行（避免上帝视角）
- 现金年化：默认 2%（可改），未持仓的现金会计息
- 换仓费率：默认 10bps（0.10%），买入/卖出各收一次

策略（右侧买入）：
- 入场：价格 > MA(ma_weeks) 且 MA 斜率为正（MA本周>上周） 且 mom_weeks 动量 > 0
- 出场：满足其一即清仓
  - 价格跌破 MA(ma_weeks)（趋势破坏）
  - 从持仓以来最高收盘回撤 >= dd（移动止盈/止损）

优化目标（仅对 dd 网格搜索）：
1) 正收益年数最大（年度收益率>0）
2) 平均收益率最大
3) 最差年份收益率最大（越不亏越好）

输出：每个自然年“当年投入”的年末收益率表。
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


def score(results: List[YearResult]) -> Tuple[int, float, float]:
    if not results:
        return (0, -1e9, -1e9)
    r = np.array([x.simple_return for x in results], dtype=float)
    return (int((r > 0).sum()), float(r.mean()), float(r.min()))


def simulate_year(
    year: int,
    weekly: pd.Series,
    ma: pd.Series,
    ma_slope_pos: pd.Series,
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
        slope_ok = ma_slope_pos.get(sig_t)
        mom_t = mom.get(sig_t)

        in_trend = (
            ma_t is not None
            and slope_ok is not None
            and mom_t is not None
            and not np.isnan(ma_t)
            and bool(slope_ok)
            and not np.isnan(mom_t)
            and px > float(ma_t)
            and float(mom_t) > 0.0
        )

        if shares > 0:
            peak = px if peak is None else max(peak, px)
            # 出场：趋势破坏 or 回撤止盈
            trend_break = ma_t is not None and not np.isnan(ma_t) and px <= float(ma_t)
            dd_hit = peak is not None and px <= peak * (1.0 - dd)
            if trend_break or dd_hit:
                sell_amt = shares * px
                cash += sell_amt * (1.0 - fee_rate)
                shares = 0.0
                peak = None

        if shares == 0 and in_trend and cash > 0:
            # 右侧入场：把现金全部买入
            buy_amt = cash * (1.0 - fee_rate)
            shares = buy_amt / px
            cash = 0.0
            peak = px
        elif shares > 0 and in_trend and cash > 0:
            # 趋势内继续投入本周现金
            buy_amt = cash * (1.0 - fee_rate)
            shares += buy_amt / px
            cash = 0.0

    # 年末估值
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


def run(
    weekly: pd.Series,
    start_year: int,
    end_year: int,
    yearly_cash: float,
    ma_weeks: int,
    mom_weeks: int,
    dd: float,
    cash_rate_annual: float,
    fee_rate: float,
    signal_lag_weeks: int,
) -> List[YearResult]:
    ma = weekly.rolling(ma_weeks, min_periods=ma_weeks).mean()
    ma_slope_pos = ma.diff() > 0
    mom = weekly / weekly.shift(mom_weeks) - 1.0
    results: List[YearResult] = []
    for y in range(start_year, end_year + 1):
        r = simulate_year(
            year=y,
            weekly=weekly,
            ma=ma,
            ma_slope_pos=ma_slope_pos,
            mom=mom,
            dd=dd,
            yearly_cash=yearly_cash,
            cash_rate_annual=cash_rate_annual,
            fee_rate=fee_rate,
            signal_lag_weeks=signal_lag_weeks,
        )
        if r is not None:
            results.append(r)
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--yearly-cash", type=float, default=200000.0)
    p.add_argument("--cash-rate-annual", type=float, default=0.02)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--signal-lag-weeks", type=int, default=1)
    p.add_argument("--ma-weeks", type=int, default=40)
    p.add_argument("--mom-weeks", type=int, default=52)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    secid = "1.510300"  # 沪深300ETF
    beg = f"{args.start_year}0101"
    end = f"{args.end_year}1231"
    weekly = to_weekly_last(fetch_daily_close(secid=secid, beg=beg, end=end))

    # 仅优化 dd
    best_dd = None
    best_results: List[YearResult] = []
    best_score = None
    for dd in [x / 100.0 for x in range(8, 26)]:  # 8%~25%
        res = run(
            weekly=weekly,
            start_year=args.start_year,
            end_year=args.end_year,
            yearly_cash=args.yearly_cash,
            ma_weeks=args.ma_weeks,
            mom_weeks=args.mom_weeks,
            dd=dd,
            cash_rate_annual=args.cash_rate_annual,
            fee_rate=args.fee_rate,
            signal_lag_weeks=args.signal_lag_weeks,
        )
        sc = score(res)
        if best_score is None or sc > best_score:
            best_score = sc
            best_dd = dd
            best_results = res

    summ = summarize(best_results)
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
            for r in best_results
        ]
    )
    content = "\n".join(
        [
            "### 沪深300ETF（510300）——右侧买入 + 回撤止盈卖出（dd自动寻优）",
            "",
            f"**入场（右侧）**：收盘>MA({args.ma_weeks}周) 且 MA斜率为正 且 动量({args.mom_weeks}周)>0（信号滞后{args.signal_lag_weeks}周执行）",
            f"**出场**：跌破MA({args.ma_weeks}周) 或 从峰值回撤>=dd",
            f"**现金年化**：{args.cash_rate_annual:.2%}；**费率**：{args.fee_rate:.2%}（买/卖各一次）",
            f"**最优dd**：{best_dd:.0%}" if best_dd is not None else "**最优dd**：-",
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


