from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


ETF_PRESETS: Dict[str, Tuple[str, str]] = {
    "hs300_etf": ("1.510300", "沪深300ETF（510300）"),
    "treasury10y_etf": ("1.511260", "十年国债ETF（511260）"),
    "gold_etf": ("1.518880", "黄金ETF（518880）"),
    "csi2000_etf": ("0.159531", "中证2000ETF（159531）"),
    "csi1000_etf": ("0.159845", "中证1000ETF（159845）"),
    "hstech_etf": ("1.513130", "恒生科技ETF（513130）"),
    "nasdaq100_etf": ("1.513100", "纳指ETF（513100）"),
    "hk_dividend_etf": ("1.513820", "港股红利ETF（513820）"),
}


@dataclass(frozen=True)
class RotationParams:
    mom_weeks: int = 52
    ma_weeks: int = 13
    abs_filter: str = "ma_only"  # ma_only/mom_only/ma_and_mom/top_mom_pos
    signal_lag_weeks: int = 1
    trailing_stop_dd: Optional[float] = 0.10
    fee_rate: float = 0.001
    cash_rate_annual: float = 0.02


@dataclass(frozen=True)
class YearlyInflowRow:
    year: int
    weeks: int
    invested: float
    end_value: float
    simple_return: float


@dataclass(frozen=True)
class CompoundRow:
    year: int
    start_value: float
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


def build_weekly_prices(universe: Dict[str, Tuple[str, str]], start_year: int, end_year: int) -> pd.DataFrame:
    beg = f"{start_year}0101"
    end = f"{end_year}1231"
    series = {}
    for key, (secid, _name) in universe.items():
        series[key] = to_weekly_last(fetch_daily_close(secid=secid, beg=beg, end=end))
    return pd.DataFrame(series).sort_index()


def compute_rotation_signal_series(weekly_prices: pd.DataFrame, p: RotationParams) -> pd.Series:
    wp = weekly_prices.sort_index()
    mom = wp / wp.shift(p.mom_weeks) - 1.0
    ma = wp.rolling(p.ma_weeks, min_periods=p.ma_weeks).mean()

    if p.abs_filter == "ma_only":
        ok = wp > ma
    elif p.abs_filter == "mom_only":
        ok = mom > 0
    elif p.abs_filter == "ma_and_mom":
        ok = (wp > ma) & (mom > 0)
    elif p.abs_filter == "top_mom_pos":
        ok = pd.DataFrame(True, index=wp.index, columns=wp.columns)
    else:
        raise ValueError("abs_filter must be one of: ma_only, mom_only, ma_and_mom, top_mom_pos")

    chosen = []
    for idx in wp.index:
        row_mom = mom.loc[idx]
        row_ok = ok.loc[idx]
        candidates = row_mom[row_ok].dropna()
        if candidates.empty:
            chosen.append(None)
        else:
            best = str(candidates.idxmax())
            if p.abs_filter == "top_mom_pos" and float(row_mom.get(best, np.nan)) <= 0:
                chosen.append(None)
            else:
                chosen.append(best)
    return pd.Series(chosen, index=wp.index, name="asset")


def latest_rotation_signal(weekly_prices: pd.DataFrame, p: RotationParams) -> Tuple[pd.Timestamp, pd.Timestamp, Optional[str], int, str, pd.DataFrame]:
    """
    返回：asof_week, exec_week, target_key(None=现金), risk_score(0/100), reason, info_table
    """
    wp = weekly_prices.dropna(how="all").sort_index()
    exec_week = wp.index[-1]
    asof_week = exec_week - pd.Timedelta(weeks=p.signal_lag_weeks)
    if asof_week not in wp.index:
        asof_week = wp.index[-(1 + p.signal_lag_weeks)]

    sig_series = compute_rotation_signal_series(wp, p)
    target = sig_series.get(asof_week, None)

    mom = wp / wp.shift(p.mom_weeks) - 1.0
    ma = wp.rolling(p.ma_weeks, min_periods=p.ma_weeks).mean()
    row_px = wp.loc[asof_week]
    row_ma = ma.loc[asof_week]
    row_mom = mom.loc[asof_week]

    if p.abs_filter == "ma_only":
        ok = (row_px > row_ma).fillna(False)
    elif p.abs_filter == "mom_only":
        ok = (row_mom > 0).fillna(False)
    elif p.abs_filter == "ma_and_mom":
        ok = ((row_px > row_ma) & (row_mom > 0)).fillna(False)
    else:
        ok = pd.Series(True, index=row_px.index)

    candidates = row_mom[ok].dropna()
    if candidates.empty:
        target = None
        reason = "无资产满足绝对过滤"
    else:
        reason = "选取动量最高者"

    dd_now = np.nan
    if target is not None and p.trailing_stop_dd is not None:
        s = wp[target].dropna()
        peak = float(s.loc[:asof_week].max())
        cur = float(s.loc[asof_week])
        dd_now = (peak - cur) / peak if peak > 0 else np.nan
        if not np.isnan(dd_now) and dd_now >= float(p.trailing_stop_dd):
            target = None
            reason = f"触发移动止损：回撤{dd_now:.0%}>= {p.trailing_stop_dd:.0%}"

    risk_score = 0 if target is None else 100

    # info table
    dds = {}
    for k in wp.columns:
        s = wp[k].dropna()
        peak = float(s.loc[:asof_week].max())
        cur = float(s.loc[asof_week]) if asof_week in s.index else np.nan
        dds[k] = (peak - cur) / peak if peak and not np.isnan(cur) else np.nan
    info = pd.DataFrame(
        {
            "价格": row_px,
            f"MA({p.ma_weeks}周)": row_ma,
            "在过滤上方": ok,
            f"动量({p.mom_weeks}周)": row_mom,
            "回撤(相对历史峰值)": pd.Series(dds),
        }
    ).sort_values(["在过滤上方", f"动量({p.mom_weeks}周)"], ascending=[False, False])

    return asof_week, exec_week, target, risk_score, reason, info


def backtest_rotation_yearly_inflow(
    weekly_prices: pd.DataFrame,
    p: RotationParams,
    start_year: int,
    end_year: int,
    yearly_cash: float,
) -> pd.DataFrame:
    """
    口径：每年投入固定 yearly_cash，按周等额入金；输出每年“当年投入”的年末收益率。
    """
    wp = weekly_prices.sort_index()
    sig = compute_rotation_signal_series(wp, p)
    cash_r_w = (1.0 + float(p.cash_rate_annual)) ** (1.0 / 52.0) - 1.0

    rows: List[YearlyInflowRow] = []
    for y in range(int(start_year), int(end_year) + 1):
        widx = wp.index[wp.index.year == y]
        if len(widx) < 45:
            continue

        deposit = float(yearly_cash) / float(len(widx))
        cash = 0.0
        shares = 0.0
        holding: Optional[str] = None
        peak_price: Optional[float] = None

        for d in widx:
            if cash > 0:
                cash *= 1.0 + cash_r_w
            cash += deposit

            # 目标信号（滞后）
            sig_t = d - pd.Timedelta(weeks=int(p.signal_lag_weeks))
            target = sig.get(sig_t, None)

            # 价格检查
            if holding is not None:
                px_h = wp.loc[d, holding]
                if pd.isna(px_h) or px_h <= 0:
                    continue
                px_h = float(px_h)
            else:
                px_h = None

            if target is not None:
                px_t = wp.loc[d, target]
                if pd.isna(px_t) or px_t <= 0:
                    continue
                px_t = float(px_t)
            else:
                px_t = None

            # 移动止损（持仓回撤）
            if holding is not None and shares > 0 and px_h is not None and p.trailing_stop_dd is not None:
                peak_price = px_h if peak_price is None else max(peak_price, px_h)
                if peak_price is not None and px_h <= peak_price * (1.0 - float(p.trailing_stop_dd)):
                    sell_amt = shares * px_h
                    cash += sell_amt * (1.0 - float(p.fee_rate))
                    shares = 0.0
                    holding = None
                    peak_price = None
                    continue

            # 若目标不变：用本周新增现金买入当前持仓
            if target == holding:
                if holding is not None and cash > 0 and px_h is not None:
                    buy_amt = cash * (1.0 - float(p.fee_rate))
                    shares += buy_amt / px_h
                    cash = 0.0
                continue

            # 换仓：先卖后买
            if holding is not None and shares > 0 and px_h is not None:
                sell_amt = shares * px_h
                cash += sell_amt * (1.0 - float(p.fee_rate))
                shares = 0.0
                holding = None
                peak_price = None

            if target is not None and cash > 0 and px_t is not None:
                buy_amt = cash * (1.0 - float(p.fee_rate))
                shares = buy_amt / px_t
                cash = 0.0
                holding = target
                peak_price = px_t

        # 年末估值
        if cash > 0:
            cash *= 1.0 + cash_r_w
        end_value = float(cash)
        if holding is not None and shares > 0:
            px_end = wp.loc[widx[-1], holding]
            if pd.notna(px_end) and px_end > 0:
                end_value += float(shares * float(px_end))

        rows.append(
            YearlyInflowRow(
                year=y,
                weeks=int(len(widx)),
                invested=float(yearly_cash),
                end_value=float(end_value),
                simple_return=float((end_value - float(yearly_cash)) / float(yearly_cash)),
            )
        )

    df = pd.DataFrame(
        [
            {
                "年份": r.year,
                "周数": r.weeks,
                "投入(元)": round(r.invested, 2),
                "年末市值(元)": round(r.end_value, 2),
                "当年收益率": r.simple_return,
            }
            for r in rows
        ]
    )
    return df


def backtest_rotation_compound(
    weekly_prices: pd.DataFrame,
    p: RotationParams,
    start_year: int,
    end_year: int,
    initial_capital: float,
    annual_contrib: float,
) -> pd.DataFrame:
    """
    口径：期初资金复利滚动；可选每年年初追加 annual_contrib；输出逐年“组合年度收益率”。
    """
    wp = weekly_prices.sort_index()
    sig = compute_rotation_signal_series(wp, p)
    cash_r_w = (1.0 + float(p.cash_rate_annual)) ** (1.0 / 52.0) - 1.0

    widx = wp.index[(wp.index.year >= int(start_year)) & (wp.index.year <= int(end_year))]
    if widx.empty:
        return pd.DataFrame(columns=["年份", "年初净值(元)", "年末净值(元)", "年度收益率"])

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

    rows: List[CompoundRow] = []
    cur_year = int(widx[0].year)
    year_start_value = value_at(widx[0])
    prev_d = widx[0]

    for d in widx:
        if int(d.year) != cur_year:
            year_end_value = value_at(prev_d)
            rows.append(
                CompoundRow(
                    year=cur_year,
                    start_value=float(year_start_value),
                    end_value=float(year_end_value),
                    simple_return=float((year_end_value - year_start_value) / year_start_value if year_start_value > 0 else 0.0),
                )
            )
            cur_year = int(d.year)
            if float(annual_contrib) > 0:
                cash += float(annual_contrib)
            year_start_value = value_at(d)

        # 现金计息
        if cash > 0:
            cash *= 1.0 + cash_r_w

        # 目标（滞后）
        sig_t = d - pd.Timedelta(weeks=int(p.signal_lag_weeks))
        target = sig.get(sig_t, None)

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

        # 移动止损
        if holding is not None and shares > 0 and px_h is not None and p.trailing_stop_dd is not None:
            peak_price = px_h if peak_price is None else max(peak_price, px_h)
            if peak_price is not None and px_h <= peak_price * (1.0 - float(p.trailing_stop_dd)):
                sell_amt = shares * px_h
                cash += sell_amt * (1.0 - float(p.fee_rate))
                shares = 0.0
                holding = None
                peak_price = None
                prev_d = d
                continue

        if target == holding:
            prev_d = d
            continue

        # 卖出
        if holding is not None and shares > 0 and px_h is not None:
            sell_amt = shares * px_h
            cash += sell_amt * (1.0 - float(p.fee_rate))
            shares = 0.0
            holding = None
            peak_price = None

        # 买入
        if target is not None and cash > 0 and px_t is not None:
            buy_amt = cash * (1.0 - float(p.fee_rate))
            shares = buy_amt / px_t
            cash = 0.0
            holding = target
            peak_price = px_t

        prev_d = d

    # 最后一年
    year_end_value = value_at(widx[-1])
    rows.append(
        CompoundRow(
            year=cur_year,
            start_value=float(year_start_value),
            end_value=float(year_end_value),
            simple_return=float((year_end_value - year_start_value) / year_start_value if year_start_value > 0 else 0.0),
        )
    )

    df = pd.DataFrame(
        [
            {
                "年份": r.year,
                "年初净值(元)": round(r.start_value, 2),
                "年末净值(元)": round(r.end_value, 2),
                "年度收益率": r.simple_return,
            }
            for r in rows
        ]
    )
    return df


