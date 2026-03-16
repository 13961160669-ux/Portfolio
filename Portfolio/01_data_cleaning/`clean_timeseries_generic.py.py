"""
背景说明：
- 应用版：在导师指导下完成，用于真实金融数据清洗与建模验证
- 本脚本：为项目申请整理的通用数据清洗模板，保留核心
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

REQUEST_START = "2010-01-01"
REQUEST_END   = "2026-03-14"

SIM_START = "2018-01-01"
SIM_END   = "2026-03-14"

PX = "close"
USE_SIM_DATA = True

def log(msg):
    print(msg, flush=True)

def build_paths():

    root = Path(__file__).resolve().parent
    raw_dir = root / "data_cache" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def generate_sim_data(start, end):
    full_dates = pd.date_range(start, end, freq="D")
    business_dates = pd.bdate_range(start, end, freq="B")

    np.random.seed(42)
    base = 6.5
    trend = np.linspace(0, 0.6, len(business_dates))
    cycle = 0.12 * np.sin(np.linspace(0, 6 * np.pi, len(business_dates)))
    noise = np.random.normal(0, 0.015, len(business_dates))
    real_rates = (base + trend + cycle + noise).round(4)

    df_biz = pd.DataFrame({PX: real_rates}, index=business_dates)
    df = df_biz.reindex(full_dates)
    df.index.name = "date"
    return df

def clean_df(df):
    n_raw = len(df)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.isna()]


    df = df[~df.index.duplicated(keep="first")].sort_index()
    n_dedup = len(df)

    if not df.index.is_monotonic_increasing:
        raise ValueError("日期非递增，请检查数据")


    df = df[df.index.weekday < 5]
    n_week = len(df)


    df[PX] = pd.to_numeric(df[PX], errors="coerce")
    df["is_invalid_price"] = (df[PX] <= 0) | df[PX].isna()
    df.loc[df["is_invalid_price"], PX] = np.nan
    n_invalid = int(df["is_invalid_price"].sum())


    df_valid = df[df[PX].notna()].copy()
    df_valid["log_return"] = np.log(df_valid[PX]).diff().replace([-np.inf, np.inf], np.nan)
    df["log_return"] = np.nan
    df.loc[df_valid.index, "log_return"] = df_valid["log_return"]


    df["is_outlier_3sigma"] = False
    ret_mask = df["log_return"].notna()
    n_ret_used = int(ret_mask.sum())

    mu = sd = lower = upper = None
    if n_ret_used > 0:
        mu = df.loc[ret_mask, "log_return"].mean()
        sd = df.loc[ret_mask, "log_return"].std()
        if pd.notna(sd) and sd > 0:
            lower = mu - 3 * sd
            upper = mu + 3 * sd
            out_mask = ret_mask & ~df["log_return"].between(lower, upper)
            df.loc[out_mask, "is_outlier_3sigma"] = True

    outlier_count = int(df["is_outlier_3sigma"].sum())
    dropped = df[df["is_outlier_3sigma"]].copy()


    if not dropped.empty:
        dropped["mu"] = mu
        dropped["sd"] = sd
        dropped["lower"] = lower
        dropped["upper"] = upper


    df_clean = df[~df["is_outlier_3sigma"]].copy()
    df_clean_valid = df_clean[df_clean[PX].notna()].copy()
    df_clean["log_return"] = np.nan
    df_clean.loc[df_clean_valid.index, "log_return"] = (
        np.log(df_clean_valid[PX]).diff().replace([-np.inf, np.inf], np.nan)
    )
    n_final = len(df_clean)


    cal_start = str(df.index.min())
    cal_end = str(df.index.max())
    px_idx = df.index[df[PX].notna()]
    px_start = str(px_idx.min()) if len(px_idx) else None
    px_end = str(px_idx.max()) if len(px_idx) else None

    meta = {
        "request_start": REQUEST_START,
        "request_end": REQUEST_END,
        "data_type": "simulated" if USE_SIM_DATA else "real",
        "sim_start": SIM_START if USE_SIM_DATA else None,
        "sim_end": SIM_END if USE_SIM_DATA else None,
        "actual_calendar_start": cal_start,
        "actual_calendar_end": cal_end,
        "actual_price_start": px_start,
        "actual_price_end": px_end,
        "request_vs_actual_warning": None,
        "n_raw": n_raw,
        "n_dedup": n_dedup,
        "n_week": n_week,
        "n_invalid": n_invalid,
        "n_returns_used_for_sigma": n_ret_used,  # 修复6
        "n_outlier": outlier_count,
        "n_final": n_final,
        "method_limitations": [
            "仅用 weekday<5 近似交易日，未处理法定节假日、临时休市、半日市",
            "真实金融数据建议对接交易所官方交易日历再对齐缺口",
            "3σ 为基础异常检测，极端行情或厚尾分布下可进一步优化"
        ],
        "note": "通用清洗模板，可直接迁移至Wind真实数据"
    }
    return df_clean, dropped, meta

def main():
    raw_dir = build_paths()


    if USE_SIM_DATA:
        df = generate_sim_data(SIM_START, SIM_END)
    else:
        raise NotImplementedError("请在此处接入真实数据读取逻辑")


    req_ts_start = pd.Timestamp(REQUEST_START)
    req_ts_end   = pd.Timestamp(REQUEST_END)
    df = df.loc[req_ts_start:req_ts_end]


    data_tag = "sim" if USE_SIM_DATA else "real"
    sdate = df.index.min().date()
    edate = df.index.max().date()
    cache_name = f"usdcny_close_{data_tag}_{sdate}_{edate}.parquet"
    cache_path = raw_dir / cache_name
    df.to_parquet(cache_path)

    log(f"缓存路径: {cache_path}")
    log(f"数据范围: {df.index.min()} ~ {df.index.max()}")


    df_clean_tmp, _, _ = clean_df(df)
    px_start = df_clean_tmp.index.min() if not df_clean_tmp.empty else None

    warn_start = (px_start is not None) and (px_start > req_ts_start)
    warn_end   = (df.index.max() < req_ts_end)

    if warn_start:
        log(f"[提示] 有效价格起始晚于请求起始: {px_start.date()} > {req_ts_start.date()}")
    if warn_end:
        log(f"[提示] 实际数据结束早于请求结束: {df.index.max().date()} < {req_ts_end.date()}")


    df_clean, dropped, meta = clean_df(df)
    meta["request_vs_actual_warning"] = bool(warn_start or warn_end)


    if not dropped.empty:
        dropped = dropped.reset_index()
        dropped["reason"] = "outlier_3sigma"
        dropped.to_csv(raw_dir / "dropped_rows.csv", index=False, encoding="utf-8-sig")


    with open(raw_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


    df_clean.to_csv(raw_dir / "usdcny_clean.csv", encoding="utf-8-sig")
    df_clean.to_parquet(raw_dir / "usdcny_clean.parquet")

    log("\n=== 清洗完成 ===")
    log(f"原始样本: {meta['n_raw']}")
    log(f"最终样本: {meta['n_final']}")
    log(f"剔除异常: {meta['n_outlier']}")

if __name__ == "__main__":
    main()