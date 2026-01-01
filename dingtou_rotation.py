#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多资产轮动（Dual Momentum/趋势轮动）：
- 用同币种（RMB计价）的ETF做比较，避免跨币种指数直接比价的失真
- 周频：用每周最后一个交易日收盘价（W-FRI last）
- 规则（默认）：
  1) 绝对动量/趋势过滤：当周收盘 > MA(40周) 且 12个月动量>0 才允许“持有风险资产”
  2) 相对动量：在候选池里选 12个月动量最高的ETF
  3) 不满足(1)则持有现金（现金年化默认2%）
  4) 换仓成本：每次从A切到B（或进出现金）收取单边费率（默认10bps=0.10%）

输出口径：仍然是“每年20万、按周等额投入，该年投入在年末的收益率”。
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

# RMB计价可交易ETF池（可按需增删）
ETF_PRESETS: Dict[str, Tuple[str, str]] = {
    "hs300_etf": ("1.510300", "沪深300ETF（510300）"),
    "csi1000_etf": ("0.159845", "中证1000ETF（159845）"),
    "csi2000_etf": ("0.159531", "中证2000ETF（159531）"),
    "hstech_etf": ("1.513130", "恒生科技ETF（513130）"),
    "nasdaq100_etf": ("1.513100", "纳指ETF（513100，跟踪纳斯达克100）"),
    "hk_dividend_etf": ("1.513820", "港股红利ETF（513820）"),
    # 股/债/金（同币种ETF，适合做跨资产轮动）
    "gold_etf": ("1.518880", "黄金ETF（518880）"),
    "treasury10y_etf": ("1.511260", "十年国债ETF（511260）"),
}


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
    rows: List[Tuple[pd.Timestamp, float]] = []
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


def build_weekly_prices(universe: Dict[str, Tuple[str, str]], beg: str, end: str) -> pd.DataFrame:
    series = {}
    for key, (secid, _name) in universe.items():
        s = fetch_daily_close(secid=secid, beg=beg, end=end)
        series[key] = to_weekly_last(s)
    prices = pd.DataFrame(series).sort_index()
    return prices


def compute_signal(
    weekly_prices: pd.DataFrame,
    mom_weeks: int = 52,
    ma_weeks: int = 40,
    abs_filter: str = "ma_and_mom",
) -> pd.Series:
    """
    返回每周选择的资产key；若不满足风控则为 None（现金）。
    采用当周收盘计算信号并以当周收盘换仓（简化假设）。
    """
    mom = weekly_prices / weekly_prices.shift(mom_weeks) - 1.0
    ma = weekly_prices.rolling(ma_weeks, min_periods=ma_weeks).mean()
    if abs_filter == "ma_only":
        ok = weekly_prices > ma
    elif abs_filter == "mom_only":
        ok = mom > 0
    elif abs_filter == "ma_and_mom":
        ok = (weekly_prices > ma) & (mom > 0)
    elif abs_filter == "top_mom_pos":
        # 不做逐资产的绝对过滤：只要求“最终选中的那只动量>0”，更接近经典动量轮动
        ok = pd.DataFrame(True, index=weekly_prices.index, columns=weekly_prices.columns)
    else:
        raise ValueError("abs_filter must be one of: ma_only, mom_only, ma_and_mom, top_mom_pos")

    chosen: List[Optional[str]] = []
    for idx in weekly_prices.index:
        row_mom = mom.loc[idx]
        row_ok = ok.loc[idx]
        candidates = row_mom[row_ok].dropna()
        if candidates.empty:
            chosen.append(None)
        else:
            best = str(candidates.idxmax())
            if abs_filter == "top_mom_pos" and float(row_mom.get(best, np.nan)) <= 0:
                chosen.append(None)
            else:
                chosen.append(best)
    return pd.Series(chosen, index=weekly_prices.index, name="asset")


def simulate_one_year(
    year: int,
    weekly_prices: pd.DataFrame,
    signal: pd.Series,
    yearly_cash: float,
    cash_rate_annual: float,
    fee_rate: float,
    signal_lag_weeks: int = 1,
    trailing_stop_dd: Optional[float] = None,
) -> Optional[YearResult]:
    """
    仅模拟“该年投入”，年内按周入金，按 signal 轮动；年末估值。
    fee_rate：每次换仓的单边费率（例如 0.001 = 10bps），按“卖出金额”和“买入金额”各收一次。
    """
    widx = weekly_prices.index[weekly_prices.index.year == year]
    if len(widx) < 45:
        return None
    deposit = yearly_cash / len(widx)
    cash = 0.0
    shares = 0.0
    holding: Optional[str] = None
    peak_price: Optional[float] = None

    cash_r_w = (1.0 + float(cash_rate_annual)) ** (1.0 / 52.0) - 1.0

    for d in widx:
        if cash > 0:
            cash *= 1.0 + cash_r_w
        cash += deposit

        sig_t = d - pd.Timedelta(weeks=signal_lag_weeks)
        target = signal.get(sig_t, None)

        # 当前价格（若缺失则跳过本周交易/估值）
        px_h = weekly_prices.loc[d, holding] if holding is not None else np.nan
        px_t = weekly_prices.loc[d, target] if target is not None else np.nan

        # 若持仓资产本周无价格，保留持仓不操作
        if holding is not None and (pd.isna(px_h) or px_h <= 0):
            continue

        # 若目标资产本周无价格，也不操作
        if target is not None and (pd.isna(px_t) or px_t <= 0):
            continue

        if holding is not None and shares > 0:
            px_now = float(px_h)
            peak_price = px_now if peak_price is None else max(peak_price, px_now)
            if trailing_stop_dd is not None and peak_price is not None:
                if px_now <= peak_price * (1.0 - float(trailing_stop_dd)):
                    sell_amt = shares * px_now
                    cash += sell_amt * (1.0 - fee_rate)
                    shares = 0.0
                    holding = None
                    peak_price = None
                    continue

        if target == holding:
            # 继续定投到当前持仓
            if holding is not None and cash > 0:
                buy_amt = cash * (1.0 - fee_rate)
                shares += buy_amt / float(px_h)
                cash = 0.0
            continue

        # 发生换仓（包括：进出现金、或A->B）
        if holding is not None and shares > 0:
            sell_amt = shares * float(px_h)
            sell_amt_net = sell_amt * (1.0 - fee_rate)
            cash += sell_amt_net
            shares = 0.0
            holding = None
            peak_price = None

        if target is not None and cash > 0:
            buy_amt = cash * (1.0 - fee_rate)
            shares = buy_amt / float(px_t)
            cash = 0.0
            holding = target
            peak_price = float(px_t)

    # 年末估值（用年内最后一个周收盘）
    last = widx[-1]
    if cash > 0:
        cash *= 1.0 + cash_r_w
    end_value = float(cash)
    if holding is not None and shares > 0:
        px = weekly_prices.loc[last, holding]
        if not pd.isna(px) and px > 0:
            end_value += float(shares * float(px))

    return YearResult(
        year=int(year),
        weeks=int(len(widx)),
        invested=float(yearly_cash),
        end_value=float(end_value),
        simple_return=float((end_value - yearly_cash) / yearly_cash),
    )


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--yearly-cash", type=float, default=200000.0)
    p.add_argument(
        "--universe",
        type=str,
        default="hs300_etf,csi1000_etf,csi2000_etf,hstech_etf,nasdaq100_etf,hk_dividend_etf",
        help="逗号分隔ETF key（见脚本 ETF_PRESETS）",
    )
    p.add_argument("--mom-weeks", type=int, default=52)
    p.add_argument("--ma-weeks", type=int, default=40)
    p.add_argument("--abs-filter", type=str, default="ma_and_mom", help="ma_only/mom_only/ma_and_mom/top_mom_pos")
    p.add_argument("--cash-rate-annual", type=float, default=0.02)
    p.add_argument("--fee-rate", type=float, default=0.001)  # 10bps
    p.add_argument("--signal-lag-weeks", type=int, default=1, help="信号滞后周数（避免同周收盘信号同周成交）")
    p.add_argument("--trailing-stop-dd", type=float, default=-1.0, help="移动止损回撤阈值（如0.15），<0表示关闭")
    p.add_argument("--optimize", action="store_true", help="小网格搜索，优先最大化正收益年数")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    keys = [x.strip() for x in args.universe.split(",") if x.strip()]
    for k in keys:
        if k not in ETF_PRESETS:
            raise ValueError(f"不支持的 universe key: {k}，可选：{', '.join(ETF_PRESETS.keys())}")
    universe = {k: ETF_PRESETS[k] for k in keys}

    beg = f"{args.start_year}0101"
    end = f"{args.end_year}1231"
    weekly_prices = build_weekly_prices(universe=universe, beg=beg, end=end)
    trailing = None if args.trailing_stop_dd < 0 else float(args.trailing_stop_dd)

    def run_once(mom_weeks: int, ma_weeks: int, abs_filter: str, trailing_stop_dd: Optional[float]) -> Tuple[pd.Series, List[YearResult]]:
        sig = compute_signal(
            weekly_prices=weekly_prices,
            mom_weeks=mom_weeks,
            ma_weeks=ma_weeks,
            abs_filter=abs_filter,
        )
        res: List[YearResult] = []
        for y in range(args.start_year, args.end_year + 1):
            r = simulate_one_year(
                year=y,
                weekly_prices=weekly_prices,
                signal=sig,
                yearly_cash=args.yearly_cash,
                cash_rate_annual=args.cash_rate_annual,
                fee_rate=args.fee_rate,
                signal_lag_weeks=args.signal_lag_weeks,
                trailing_stop_dd=trailing_stop_dd,
            )
            if r is not None:
                res.append(r)
        return sig, res

    # 实际使用的参数（优化时会被覆盖）
    used_mom = args.mom_weeks
    used_ma = args.ma_weeks
    used_abs = args.abs_filter
    used_dd = trailing
    best_params = None
    if args.optimize:
        def score(res: List[YearResult]) -> Tuple[int, float, float]:
            if not res:
                return (0, -1e9, -1e9)
            r = np.array([x.simple_return for x in res], dtype=float)
            return (int((r > 0).sum()), float(r.mean()), float(r.min()))

        best_s = None
        best = None
        for mom in (13, 26, 52):
            for ma in (13, 20, 40, 52):
                for af in ("ma_only", "mom_only", "ma_and_mom", "top_mom_pos"):
                    for dd in (None, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
                        sig, res = run_once(mom_weeks=mom, ma_weeks=ma, abs_filter=af, trailing_stop_dd=dd)
                        s = score(res)
                        if best_s is None or s > best_s:
                            best_s = s
                            best = (sig, res)
                            best_params = (mom, ma, af, dd)
        signal, results = best if best is not None else (pd.Series(dtype=object), [])
        if best_params is not None:
            used_mom, used_ma, used_abs, used_dd = best_params
    else:
        signal, results = run_once(
            mom_weeks=args.mom_weeks,
            ma_weeks=args.ma_weeks,
            abs_filter=args.abs_filter,
            trailing_stop_dd=trailing,
        )

    summ = summarize(results)
    summ_line = "；".join([f"{k}={v}" for k, v in summ.items()])
    universe_line = "、".join([universe[k][1] for k in keys])
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
            "### 多资产轮动（Dual Momentum）——在多个指数/ETF之间切换以规避单一指数大熊市",
            "",
            f"**ETF池**：{universe_line}",
            f"**绝对过滤**：abs_filter={used_abs}；MA={used_ma}周；动量窗口={used_mom}周",
            f"**相对动量**：在满足条件的资产中选 过去{used_mom}周动量最高者",
            f"**信号滞后**：{args.signal_lag_weeks}周；**现金年化**：{args.cash_rate_annual:.2%}；**换仓费率**：{args.fee_rate:.2%}（卖出/买入各一次）",
            (f"**移动止损**：回撤{float(used_dd):.0%}" if used_dd is not None else "**移动止损**：关闭"),
            (f"**已优化参数**：mom={best_params[0]}周；ma={best_params[1]}周；abs_filter={best_params[2]}；dd={best_params[3]}" if args.optimize and best_params else ""),
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


