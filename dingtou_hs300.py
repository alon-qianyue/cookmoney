#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每年20万，按周等额定投指数（默认：沪深300）：计算过去10年（默认 2016-2025）每年当年投入的收益率。

口径（默认）：
- 标的：指数（默认沪深300；也支持中证1000）
- 数据：东方财富 push2his kline（日线前复权/指数不受复权影响）
- 买入时点：每周一次，使用该周最后一个交易日的收盘价（W-FRI 周频取 last）
- 年度资金：每年总投入 200,000 元，平均分配到该年出现的周买点（周数可能为 52/53）
- 年度收益率：当年投入形成的份额，在当年最后一个交易日收盘价估值
  年度收益率 = (年末市值 - 200000) / 200000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

INDEX_PRESETS = {
    "hs300": ("1.000300", "沪深300指数（000300）"),
    "csi1000": ("1.000852", "中证1000指数（000852）"),
    "csi_dividend": ("1.000922", "中证红利指数（000922）"),
    # 东方财富 suggest 返回 QuoteID=2.932000
    "csi2000": ("2.932000", "中证2000指数（932000）"),
    # 港股/恒生系列（QuoteID 来自东方财富 suggest）
    "hstech": ("124.HSTECH", "恒生科技指数（HSTECH）"),
    "hs_high_div_yield": ("124.HSHDYI", "恒生高股息率指数（HSHDYI）"),
    "stock_connect_high_div": ("2.930914", "港股通高股息指数（930914）"),
    # 美股/海外指数
    "nasdaq100": ("100.NDX100", "纳斯达克100指数（NDX100）"),
    "qqq": ("105.QQQ", "QQQ（Invesco QQQ Trust）"),
}


@dataclass(frozen=True)
class YearResult:
    year: int
    weeks: int
    invested: float
    end_value: float
    simple_return: float
    strategy: str = "plain"


def fetch_index_daily(secid: str, beg: str, end: str, timeout: int = 20) -> pd.DataFrame:
    """
    返回 DataFrame: [date, close]
    date: datetime64[ns]
    close: float
    """
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",  # 日线
        "fqt": "1",
        "beg": beg,
        "end": end,
    }
    resp = requests.get(EASTMONEY_KLINE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"东方财富接口无数据：{json.dumps(payload, ensure_ascii=False)[:500]}")

    rows: List[Tuple[pd.Timestamp, float]] = []
    for k in klines:
        # "2025-12-31,xxxx(close),..."
        parts = k.split(",")
        if len(parts) < 3:
            continue
        d = pd.to_datetime(parts[0])
        close = float(parts[2])
        rows.append((d, close))

    df = pd.DataFrame(rows, columns=["date", "close"]).dropna()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def calc_yearly_dca_returns(
    daily: pd.DataFrame, start_year: int, end_year: int, yearly_cash: float
) -> List[YearResult]:
    s = daily.set_index("date")["close"].astype(float)
    s = s[s.index.notna()].sort_index()

    # 周频：以周五为锚点，取该周最后一个交易日收盘
    weekly = s.resample("W-FRI").last().dropna()

    results: List[YearResult] = []
    for y in range(start_year, end_year + 1):
        w = weekly[(weekly.index.year == y)]
        if w.empty:
            continue
        # 有些指数在样本期内并非全年可用（例如中途发布/数据源缺失）。
        # 为避免出现“只有几周数据却按全年20万投入”的失真，这里要求至少覆盖大部分年份。
        if int(w.shape[0]) < 45:
            continue
        weeks = int(w.shape[0])
        cash_per_week = yearly_cash / weeks
        shares = (cash_per_week / w).sum()

        # 年末估值：用当年最后一个交易日收盘价（daily里该年的最后一天）
        year_daily = s[(s.index.year == y)]
        end_close = float(year_daily.iloc[-1])
        end_value = float(shares * end_close)
        simple_ret = (end_value - yearly_cash) / yearly_cash
        results.append(
            YearResult(
                year=y,
                weeks=weeks,
                invested=float(yearly_cash),
                end_value=end_value,
                simple_return=float(simple_ret),
            )
        )

    return results


def calc_yearly_trend_filter_cash(
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    yearly_cash: float,
    ma_weeks: int = 40,
) -> List[YearResult]:
    """
    趋势过滤 + 现金（现金收益按0计）：
    - 周收盘 > MA(ma_weeks)：当周投入买入指数
    - 否则：当周投入进入现金
    """
    s = daily.set_index("date")["close"].astype(float)
    s = s[s.index.notna()].sort_index()
    weekly = s.resample("W-FRI").last().dropna()
    ma = weekly.rolling(ma_weeks, min_periods=ma_weeks).mean()

    results: List[YearResult] = []
    for y in range(start_year, end_year + 1):
        w = weekly[weekly.index.year == y]
        if w.empty:
            continue
        if int(w.shape[0]) < 45:
            continue
        weeks = int(w.shape[0])
        cash_per_week = yearly_cash / weeks
        shares = 0.0
        cash_balance = 0.0

        for d, px in w.items():
            m = ma.get(d)
            if m is not None and not np.isnan(m) and float(px) > float(m):
                shares += cash_per_week / float(px)
            else:
                cash_balance += cash_per_week

        end_close = float(w.iloc[-1])
        end_value = float(shares * end_close + cash_balance)
        simple_ret = (end_value - yearly_cash) / yearly_cash
        results.append(
            YearResult(
                year=y,
                weeks=weeks,
                invested=float(yearly_cash),
                end_value=end_value,
                simple_return=float(simple_ret),
                strategy=f"trend_ma{ma_weeks}_cash",
            )
        )
    return results


def calc_yearly_drawdown_weighted_dca(
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    yearly_cash: float,
    lookback_weeks: int = 52,
) -> List[YearResult]:
    """
    回撤加权定投（当年总投入固定=yearly_cash，仍全部买指数）：
    - 基于滚动 lookback_weeks 周高点计算回撤 dd
    - 权重 = 1 + dd（越跌权重越高）
    - 当年按权重归一化分配全年资金到每周买点
    """
    s = daily.set_index("date")["close"].astype(float)
    s = s[s.index.notna()].sort_index()
    weekly = s.resample("W-FRI").last().dropna()

    roll_max = weekly.rolling(lookback_weeks, min_periods=lookback_weeks).max()
    dd = ((roll_max - weekly) / roll_max).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    weights = (1.0 + dd).clip(lower=0.1)

    results: List[YearResult] = []
    for y in range(start_year, end_year + 1):
        w = weekly[weekly.index.year == y]
        if w.empty:
            continue
        if int(w.shape[0]) < 45:
            continue
        weeks = int(w.shape[0])
        ww = weights[weights.index.year == y].reindex(w.index).fillna(1.0).clip(lower=0.1)
        alloc = yearly_cash * (ww / float(ww.sum()))
        shares = float((alloc / w).sum())

        end_close = float(w.iloc[-1])
        end_value = float(shares * end_close)
        simple_ret = (end_value - yearly_cash) / yearly_cash
        results.append(
            YearResult(
                year=y,
                weeks=weeks,
                invested=float(yearly_cash),
                end_value=end_value,
                simple_return=float(simple_ret),
                strategy=f"drawdown_w{lookback_weeks}",
            )
        )
    return results


def summarize(results: List[YearResult]) -> Dict[str, str]:
    if not results:
        return {
            "样本年数": "0",
            "正收益年数": "0",
            "亏损年数": "0",
            "最差年份": "-",
            "最差收益率": "-",
            "平均收益率": "-",
            "中位数收益率": "-",
        }
    rets = np.array([r.simple_return for r in results], dtype=float)
    years = [r.year for r in results]
    worst_idx = int(np.argmin(rets))
    return {
        "样本年数": str(int(rets.size)),
        "正收益年数": str(int((rets > 0).sum())),
        "亏损年数": str(int((rets < 0).sum())),
        "最差年份": str(years[worst_idx]),
        "最差收益率": f"{rets[worst_idx] * 100:.2f}%",
        "平均收益率": f"{rets.mean() * 100:.2f}%",
        "中位数收益率": f"{np.median(rets) * 100:.2f}%",
    }


def calc_yearly_trend_vol_target(
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    yearly_cash: float,
    ma_weeks: int = 40,
    vol_lookback_weeks: int = 20,
    target_vol_annual: float = 0.15,
    leverage_max: float = 1.5,
    cash_rate_annual: float = 0.02,
    borrow_rate_annual: float = 0.03,
) -> List[YearResult]:
    """
    策略A：趋势过滤 + 波动目标仓位（允许杠杆，上限 leverage_max；借款成本按0计）
    - 趋势：周收盘 > MA(ma_weeks) 才允许持仓
    - 目标杠杆：L = min(leverage_max, target_vol / current_vol)；若vol不可用则L=1
      其中 current_vol 为近 vol_lookback_weeks 周收益率年化波动（std*sqrt(52)）
    - 执行：每周先入金（当年总计yearly_cash），然后根据目标杠杆对“总权益”做再平衡
    """
    s = daily.set_index("date")["close"].astype(float)
    s = s[s.index.notna()].sort_index()
    weekly = s.resample("W-FRI").last().dropna()
    ma = weekly.rolling(ma_weeks, min_periods=ma_weeks).mean()
    weekly_ret = weekly.pct_change()
    vol = weekly_ret.rolling(vol_lookback_weeks, min_periods=vol_lookback_weeks).std() * np.sqrt(52.0)

    results: List[YearResult] = []
    for y in range(start_year, end_year + 1):
        w = weekly[weekly.index.year == y]
        if w.empty:
            continue
        if int(w.shape[0]) < 45:
            continue

        weeks = int(w.shape[0])
        deposit = yearly_cash / weeks
        shares = 0.0
        cash = 0.0
        debt = 0.0

        cash_rate_w = (1.0 + float(cash_rate_annual)) ** (1.0 / 52.0) - 1.0
        borrow_rate_w = (1.0 + float(borrow_rate_annual)) ** (1.0 / 52.0) - 1.0

        for d, px in w.items():
            # 周度计息：现金/负债
            if cash > 0:
                cash *= 1.0 + cash_rate_w
            if debt > 0:
                debt *= 1.0 + borrow_rate_w

            cash += deposit
            px = float(px)

            m = ma.get(d)
            in_trend = m is not None and not np.isnan(m) and px > float(m)
            if not in_trend:
                # 清仓到现金
                cash += shares * px
                shares = 0.0
                # 偿还负债
                pay = min(cash, debt)
                cash -= pay
                debt -= pay
                continue

            v = vol.get(d)
            if v is None or np.isnan(v) or float(v) <= 0:
                L = 1.0
            else:
                L = float(target_vol_annual) / float(v)
            L = float(np.clip(L, 0.0, leverage_max))

            equity = shares * px + cash - debt
            # 若权益已经<=0，风险失控，直接清仓并结束该年（避免数值爆炸）
            if equity <= 0:
                cash += shares * px
                shares = 0.0
                pay = min(cash, debt)
                cash -= pay
                debt -= pay
                # 剩余周入金也只放现金
                continue

            target_exposure = equity * L
            current_exposure = shares * px
            delta = target_exposure - current_exposure

            if delta > 0:
                # 买入：先用现金，不够再加杠杆借款
                use_cash = min(cash, delta)
                cash -= use_cash
                delta -= use_cash
                if delta > 0:
                    debt += delta
                shares += (use_cash + (delta if delta > 0 else 0.0)) / px
            elif delta < 0:
                sell_amount = min(-delta, current_exposure)
                shares -= sell_amount / px
                cash += sell_amount
                # 优先还债
                pay = min(cash, debt)
                cash -= pay
                debt -= pay

        end_px = float(w.iloc[-1])
        end_value = float(shares * end_px + cash - debt)
        simple_ret = (end_value - yearly_cash) / yearly_cash
        results.append(
            YearResult(
                year=y,
                weeks=weeks,
                invested=float(yearly_cash),
                end_value=end_value,
                simple_return=float(simple_ret),
                strategy=(
                    f"trend_voltarget_ma{ma_weeks}_v{vol_lookback_weeks}"
                    f"_t{int(target_vol_annual*100)}_L{leverage_max}"
                    f"_cash{int(cash_rate_annual*100)}_borrow{int(borrow_rate_annual*100)}"
                ),
            )
        )
    return results


def calc_yearly_trend_trailing_stop(
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    yearly_cash: float,
    ma_weeks: int = 40,
    stop_dd: float = 0.15,
    cash_rate_annual: float = 0.02,
) -> List[YearResult]:
    """
    策略B：趋势过滤 + 回撤止损（移动止盈/止损）再入场（不加杠杆，现金收益按0计）
    - 仅在 周收盘 > MA(ma_weeks) 时允许持仓/买入
    - 持仓期间记录 peak_price；若价格从peak回撤 >= stop_dd：清仓到现金
    - 清仓后，直到再次满足 周收盘 > MA(ma_weeks) 才允许重新买入
    - 每周入金固定（当年总计yearly_cash），在允许持仓时把现金全部买入
    """
    s = daily.set_index("date")["close"].astype(float)
    s = s[s.index.notna()].sort_index()
    weekly = s.resample("W-FRI").last().dropna()
    ma = weekly.rolling(ma_weeks, min_periods=ma_weeks).mean()

    results: List[YearResult] = []
    for y in range(start_year, end_year + 1):
        w = weekly[weekly.index.year == y]
        if w.empty:
            continue
        if int(w.shape[0]) < 45:
            continue

        weeks = int(w.shape[0])
        deposit = yearly_cash / weeks
        shares = 0.0
        cash = 0.0
        peak: Optional[float] = None
        cash_rate_w = (1.0 + float(cash_rate_annual)) ** (1.0 / 52.0) - 1.0

        for d, px in w.items():
            if cash > 0:
                cash *= 1.0 + cash_rate_w
            cash += deposit
            px = float(px)
            m = ma.get(d)
            in_trend = m is not None and not np.isnan(m) and px > float(m)

            if shares > 0:
                peak = px if peak is None else max(peak, px)
                if peak is not None and px <= peak * (1.0 - stop_dd):
                    # 触发回撤止损：清仓
                    cash += shares * px
                    shares = 0.0
                    peak = None

            if in_trend and shares == 0 and cash > 0:
                # 重新入场：把现金全部买入
                shares = cash / px
                cash = 0.0
                peak = px
            elif in_trend and shares > 0 and cash > 0:
                # 趋势内：把本周新增现金买入（维持“持续定投”）
                shares += cash / px
                cash = 0.0

            if not in_trend and shares > 0:
                # 趋势破坏：清仓到现金
                cash += shares * px
                shares = 0.0
                peak = None

        end_px = float(w.iloc[-1])
        end_value = float(shares * end_px + cash)
        simple_ret = (end_value - yearly_cash) / yearly_cash
        results.append(
            YearResult(
                year=y,
                weeks=weeks,
                invested=float(yearly_cash),
                end_value=end_value,
                simple_return=float(simple_ret),
                strategy=f"trend_trailingstop_ma{ma_weeks}_dd{int(stop_dd*100)}_cash{int(cash_rate_annual*100)}",
            )
        )
    return results


def pick_best(results: List[YearResult]) -> Tuple[int, float, float]:
    """
    优化目标（从高到低优先级）：
    1) 正收益年数最大（>0）
    2) 平均收益率最大
    3) 最差年份收益率最大（越不亏越好）
    返回可比较的三元组。
    """
    if not results:
        return (0, -1e9, -1e9)
    rets = np.array([r.simple_return for r in results], dtype=float)
    pos_years = int((rets > 0).sum())
    mean_ret = float(rets.mean())
    worst_ret = float(rets.min())
    return (pos_years, mean_ret, worst_ret)


def optimize_two_strategies(
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    yearly_cash: float,
    cash_rate_annual: float,
    borrow_rate_annual: float,
) -> Dict[str, List[YearResult]]:
    """
    对两条策略做小规模网格搜索，返回每条策略的“最佳参数版本”的逐年结果。
    """
    # 策略A：趋势+波动目标
    best_a: Optional[List[YearResult]] = None
    best_a_score: Optional[Tuple[int, float, float]] = None
    for ma in (26, 40, 52):
        for v_lb in (13, 20, 26):
            for tvol in (0.10, 0.15, 0.20):
                for L in (1.0, 1.5, 2.0):
                    r = calc_yearly_trend_vol_target(
                        daily=daily,
                        start_year=start_year,
                        end_year=end_year,
                        yearly_cash=yearly_cash,
                        ma_weeks=ma,
                        vol_lookback_weeks=v_lb,
                        target_vol_annual=tvol,
                        leverage_max=L,
                        cash_rate_annual=cash_rate_annual,
                        borrow_rate_annual=borrow_rate_annual,
                    )
                    score = pick_best(r)
                    if best_a_score is None or score > best_a_score:
                        best_a_score = score
                        best_a = r

    # 策略B：趋势+回撤止损
    best_b: Optional[List[YearResult]] = None
    best_b_score: Optional[Tuple[int, float, float]] = None
    for ma in (26, 40, 52):
        for dd in (0.10, 0.12, 0.15, 0.18, 0.20):
            r = calc_yearly_trend_trailing_stop(
                daily=daily,
                start_year=start_year,
                end_year=end_year,
                yearly_cash=yearly_cash,
                ma_weeks=ma,
                stop_dd=dd,
                cash_rate_annual=cash_rate_annual,
            )
            score = pick_best(r)
            if best_b_score is None or score > best_b_score:
                best_b_score = score
                best_b = r

    return {
        "trend_voltarget_optimized": best_a or [],
        "trend_trailingstop_optimized": best_b or [],
    }


def to_markdown_table(results: List[YearResult]) -> str:
    df = pd.DataFrame(
        [
            {
                "年份": r.year,
                "周数": r.weeks,
                "投入(元)": round(r.invested, 2),
                "年末市值(元)": round(r.end_value, 2),
                "当年收益率": f"{r.simple_return * 100:.2f}%",
            }
            for r in results
        ]
    )
    return df.to_markdown(index=False)


def parse_indices(value: str) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("indices 不能为空")
    for x in items:
        if x not in INDEX_PRESETS:
            raise ValueError(f"不支持的 indices: {x}，可选：{', '.join(INDEX_PRESETS.keys())}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--yearly-cash", type=float, default=200000.0)
    parser.add_argument(
        "--indices",
        type=str,
        default="hs300",
        help="逗号分隔：hs300,csi1000,csi_dividend,csi2000,hstech,hs_high_div_yield,stock_connect_high_div,nasdaq100,qqq",
    )
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--strategies",
        type=str,
        default="plain",
        help="逗号分隔：plain,trend_ma40_cash,drawdown_w52,trend_voltarget_ma40,trend_trailingstop_ma40_dd15",
    )
    parser.add_argument("--cash-rate-annual", type=float, default=0.02, help="现金年化收益率（用于策略里的现金部分）")
    parser.add_argument("--borrow-rate-annual", type=float, default=0.03, help="融资年化成本（用于波动目标策略的杠杆部分）")
    parser.add_argument("--optimize-two", action="store_true", help="对两条策略做参数优化（输出最佳参数版本）")
    args = parser.parse_args()

    # 多取一点缓冲，确保覆盖周频重采样的边界
    beg = f"{args.start_year}0101"
    end = f"{args.end_year}1231"

    indices = parse_indices(args.indices)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    sections: List[str] = []
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key in indices:
        secid, display_name = INDEX_PRESETS[key]
        daily = fetch_index_daily(secid=secid, beg=beg, end=end)
        blocks: List[str] = []

        if args.optimize_two:
            best_map = optimize_two_strategies(
                daily=daily,
                start_year=args.start_year,
                end_year=args.end_year,
                yearly_cash=args.yearly_cash,
                cash_rate_annual=args.cash_rate_annual,
                borrow_rate_annual=args.borrow_rate_annual,
            )
            strategy_results = [(k, v) for k, v in best_map.items()]
        else:
            strategy_results = []
            for st in strategies:
                if st == "plain":
                    results = calc_yearly_dca_returns(
                        daily=daily,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        yearly_cash=args.yearly_cash,
                    )
                elif st == "trend_ma40_cash":
                    results = calc_yearly_trend_filter_cash(
                        daily=daily,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        yearly_cash=args.yearly_cash,
                        ma_weeks=40,
                    )
                elif st == "drawdown_w52":
                    results = calc_yearly_drawdown_weighted_dca(
                        daily=daily,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        yearly_cash=args.yearly_cash,
                        lookback_weeks=52,
                    )
                elif st == "trend_voltarget_ma40":
                    results = calc_yearly_trend_vol_target(
                        daily=daily,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        yearly_cash=args.yearly_cash,
                        ma_weeks=40,
                        vol_lookback_weeks=20,
                        target_vol_annual=0.15,
                        leverage_max=1.5,
                        cash_rate_annual=args.cash_rate_annual,
                        borrow_rate_annual=args.borrow_rate_annual,
                    )
                elif st == "trend_trailingstop_ma40_dd15":
                    results = calc_yearly_trend_trailing_stop(
                        daily=daily,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        yearly_cash=args.yearly_cash,
                        ma_weeks=40,
                        stop_dd=0.15,
                        cash_rate_annual=args.cash_rate_annual,
                    )
                else:
                    raise ValueError(f"不支持的策略：{st}")
                strategy_results.append((st, results))

        for st, results in strategy_results:
            summ = summarize(results)
            summ_line = "；".join([f"{k}={v}" for k, v in summ.items()])
            blocks.append(
                "\n".join(
                    [
                        f"#### 策略：{st}",
                        "",
                        f"**摘要**：{summ_line}",
                        "",
                        to_markdown_table(results),
                    ]
                )
            )

        title = (
            f"### {display_name}：每年20万、周度等额定投（策略对比：当年投入的年末收益率）"
            if (args.optimize_two or len(strategies) > 1)
            else f"### {display_name}：每年20万、周度等额定投（当年投入的年末收益率）"
        )
        header = "\n".join(
            [
                title,
                "",
                f"**标的**：{display_name}",
                "**买入基准**：每周一次，用该周最后一个交易日收盘价（周频 `W-FRI` 取 last）",
                "**年度资金**：每年 200,000 元",
                f"**现金年化**：{args.cash_rate_annual:.2%}；**融资年化**：{args.borrow_rate_annual:.2%}",
                "",
                f"**生成时间**：{now_str}",
                "",
            ]
        )
        sections.append(header + "\n\n".join(blocks))

    content = "\n\n".join(sections) + "\n"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


