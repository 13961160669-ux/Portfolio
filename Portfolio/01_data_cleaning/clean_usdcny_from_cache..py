from __future__ import annotations

import sys
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

START = "2010-01-01"
END = "2026-03-14"

WIND_CODE = "USDCNY.EX"
WIND_FIELD = "close"

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = RAW_DIR / f"usdcny_{WIND_CODE.replace('.', '_')}_{WIND_FIELD}_{START}_{END}.parquet"

try:
    from WindPy import w
    HAS_WIND = True
    _WIND_IMPORT_ERR = None
except Exception as e:
    w = None
    HAS_WIND = False
    _WIND_IMPORT_ERR = e


def wind_start() -> None:
    if not HAS_WIND or w is None:
        raise RuntimeError(f"WindPy import failed: {_WIND_IMPORT_ERR}")

    try:
        connected = bool(w.isconnected())
    except Exception:
        connected = False

    if not connected:
        r = w.start()
        err = getattr(r, "ErrorCode", 0)
        if err != 0:
            raise RuntimeError(f"WindPy start failed: ErrorCode={err}")


def _to_dt(s: str) -> str:
    datetime.strptime(s, "%Y-%m-%d")
    return s


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    out = df.sort_index().reset_index().rename(columns={"index": "date"})
    out.to_parquet(path, index=False)


def fetch_wind_daily(code: str, field: str, start: str, end: str) -> pd.DataFrame:
    wind_start()
    start = _to_dt(start)
    end = _to_dt(end)

    r = w.wsd(code, field, start, end, "PriceAdj=F")
    err = getattr(r, "ErrorCode", 0)
    if err != 0:
        raise RuntimeError(f"w.wsd failed: ErrorCode={err}")

    times = getattr(r, "Times", None)
    data = getattr(r, "Data", None)
    if not times or data is None:
        raise RuntimeError("w.wsd returned empty data")

    values = data[0] if isinstance(data, list) and len(data) > 0 else data
    df = pd.DataFrame({field: values}, index=pd.to_datetime(times))
    df.index.name = "date"
    df[field] = pd.to_numeric(df[field], errors="coerce")
    return df.sort_index()


def fetch_daily_with_cache(code: str, field: str, start: str, end: str, cache_path: Path) -> pd.DataFrame:
    df_cache = _load_cache(cache_path)
    if df_cache is not None and len(df_cache) > 0:
        print(f"[INFO] 成功从本地缓存加载数据: {cache_path.name}")
        if field not in df_cache.columns and len(df_cache.columns) == 1:
            df_cache = df_cache.rename(columns={df_cache.columns[0]: field})
        return df_cache

    print(f"[INFO] 本地缓存不存在，尝试从 Wind 获取数据...")
    try:
        df = fetch_wind_daily(code, field, start, end)
        _save_cache(df, cache_path)
        print(f"[INFO] 从 Wind 获取成功，并已保存至缓存")
        return df
    except Exception as e:
        raise RuntimeError(
            f"数据获取失败：\n"
            f"1. 本地无缓存文件\n"
            f"2. Wind 连接失败: {e}\n"
            f"请先获取真实数据并放入缓存路径，或在 Wind 环境下运行程序。"
        ) from e


def main() -> None:
    print("Python 解释器路径:", sys.executable)
    print("缓存文件路径:", str(CACHE_PATH), "\n")

    df = fetch_daily_with_cache(WIND_CODE, WIND_FIELD, START, END, CACHE_PATH)

    print("=" * 60)
    print("数据预览（前5行）:")
    print(df.head())
    print("\n数据预览（后5行）:")
    print(df.tail())
    print("=" * 60)
    print(f"总样本量: {len(df)} 行")
    print(f"时间区间: {df.index.min()}  ~  {df.index.max()}")

    px = WIND_FIELD

    # 1. 去重 + 排序
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # 2. 保留工作日
    before_weekday = len(df)
    df = df[df.index.weekday < 5]
    print(f"[清洗] 工作日筛选: {before_weekday} -> {len(df)} 行")

    # 3. 价格数据标准化
    df[px] = pd.to_numeric(df[px], errors="coerce")
    df.loc[df[px] <= 0, px] = np.nan

    # 4. 计算对数收益率
    df["log_return"] = np.log(df[px]).diff()
    df["log_return"] = df["log_return"].replace([np.inf, -np.inf], np.nan)

    # 4. 对收益率做 3σ 异常值剔除
    mask = df["log_return"].notna()
    mu = df.loc[mask, "log_return"].mean()
    sd = df.loc[mask, "log_return"].std()
    df = df[~mask | df["log_return"].between(mu - 3 * sd, mu + 3 * sd)]

    # 5. 数据质量统计
    fx_series = df[px].rename("fx_mid")
    print(f"有效价格数据量: {fx_series.dropna().shape[0]} 行")
    print(f"价格缺失值数量: {fx_series.isna().sum()} 个")

    # 6. 统计
    print("\n=== 描述性统计 ===")
    print(df[[px, "log_return"]].dropna().describe())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[TRACEBACK]")
        traceback.print_exc()
        sys.exit(1)