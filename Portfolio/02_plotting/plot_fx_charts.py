"""
Drug Release Kinetics Fitting Pipeline
For Wound Dressing / Hydrogel Research
RA Portfolio — Safe, Robust, No Pitfalls
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Optional


mpl.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False


SEED = 42
np.random.seed(SEED)
OUT = Path("output")
OUT.mkdir(exist_ok=True)


def zero_order(t, k):
    return k * t

def first_order(t, k):
    return 1.0 - np.exp(-k * t)

def higuchi(t, k):
    return k * np.sqrt(np.maximum(t, 0.0))

def korsmeyer_peppas(t, k, n):
    return k * np.power(np.maximum(t, 1e-12), n)

def clip01(y):
    return np.clip(y, 0.0, 1.0)


@dataclass
class FitResult:
    name: str
    n_param: int
    n_used: int
    rmse_train: float
    rmse_test: float
    aic: Optional[float]
    aicc: Optional[float]
    popt: np.ndarray
    is_kp_early: bool = False


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def aic_safe(y_true, y_pred, k, eps=1e-12):
    n = len(y_true)
    rss = float(np.sum((y_true - y_pred)**2))
    rss = max(rss, eps)
    return 2 * k + n * np.log(rss / n)

def aicc_safe(y_true, y_pred, k):
    n = len(y_true)
    if n <= k + 1:
        return None
    return aic_safe(y_true, y_pred, k) + (2 * k * (k + 1)) / (n - k - 1)


t = np.linspace(0, 48, 60)
k_true = 0.10
y_true = first_order(t, k_true)
noise = np.random.normal(0, 0.02, size=t.size)
burst = 0.06 * np.exp(-t/1.5)
y_obs = clip01(y_true + burst + noise)

# 时间序列切分
n_train = int(0.8 * len(t))
t_train, y_train = t[:n_train], y_obs[:n_train]
t_test, y_test = t[n_train:], y_obs[n_train:]

# KP 早期点（防不足）
mask_kp = y_train <= 0.6
t_kp, y_kp = t_train[mask_kp], y_train[mask_kp]
MIN_KP_POINTS = 10
has_enough_kp_points = len(t_kp) >= MIN_KP_POINTS


fits = []

def fit_and_store(name, model_func, t_fit, y_fit, n_param, p0, bounds, is_kp=False):
    try:
        popt, _ = curve_fit(model_func, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
    except (RuntimeError, ValueError) as e:
        print(f" {name} 拟合失败：{e}，跳过")
        return

    y_hat_train = model_func(t_train, *popt)
    y_hat_test = model_func(t_test, *popt)

    aic_val = aic_safe(y_fit, model_func(t_fit, *popt), n_param)
    aicc_val = aicc_safe(y_fit, model_func(t_fit, *popt), n_param)

    fits.append(FitResult(
        name=name,
        n_param=n_param,
        n_used=len(t_fit),
        rmse_train=rmse(y_train, y_hat_train),
        rmse_test=rmse(y_test, y_hat_test),
        aic=aic_val,
        aicc=aicc_val,
        popt=popt,
        is_kp_early=is_kp
    ))


fit_and_store("Zero-order", zero_order, t_train, y_train, 1, [0.01], (0, 1))


fit_and_store("First-order", first_order, t_train, y_train, 1, [0.05], (0, 5))


fit_and_store("Higuchi", higuchi, t_train, y_train, 1, [0.05], (0, 2))


if has_enough_kp_points:
    fit_and_store("Korsmeyer–Peppas", korsmeyer_peppas, t_kp, y_kp, 2, [0.1,0.5], ((0,0),(5,2)), is_kp=True)
else:
    print(f"释放点不足（<{MIN_KP_POINTS}），跳过 KP 模型")

fits = sorted(fits, key=lambda x: x.rmse_test)
best = fits[0] if fits else None


rows = []
for f in fits:
    rows.append({
        "model": f.name,
        "n_param": f.n_param,
        "n_used": f.n_used,
        "rmse_train": round(f.rmse_train, 4),
        "rmse_test": round(f.rmse_test, 4),
        "aic": round(f.aic, 2) if f.aic is not None else None,
        "aicc": round(f.aicc, 2) if f.aicc is not None else None,
        "kp_early_only": f.is_kp_early,
        "note": "KP仅拟合早期≤0.6，RMSE含外推误差" if f.is_kp_early else ""
    })
pd.DataFrame(rows).to_csv(OUT / "fit_results.csv", index=False, encoding="utf-8-sig")


tt = np.linspace(0,48,300)
plt.figure(figsize=(10,6))
plt.scatter(t_train, y_train, c="black", s=16, label="训练", alpha=0.85)
plt.scatter(t_test, y_test, c="crimson", s=16, label="测试", alpha=0.85)

colors = {
    "Zero-order": "#1f77b4",
    "First-order": "#ff7f0e",
    "Higuchi": "#2ca02c",
    "Korsmeyer–Peppas": "#d62728"
}

for f in fits:
    if f.name == "Zero-order":
        yy = clip01(zero_order(tt, *f.popt))
    elif f.name == "First-order":
        yy = clip01(first_order(tt, *f.popt))
    elif f.name == "Higuchi":
        yy = clip01(higuchi(tt, *f.popt))
    else:
        yy = clip01(korsmeyer_peppas(tt, *f.popt))
    plt.plot(tt, yy, label=f.name, c=colors[f.name], lw=2.2)

plt.axvline(t_train[-1], color="gray", linestyle="--", alpha=0.6, label="训练/测试分割")
plt.title("Drug Release Fitting (Train+Test)")
plt.xlabel("时间 (h)")
plt.ylabel("累积释放率 Mt/M∞")
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.2)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "best_fit.png", dpi=220, bbox_inches="tight")
plt.close()


if best:
    if best.name == "Zero-order":
        y_fit = zero_order(t_train, *best.popt)
    elif best.name == "First-order":
        y_fit = first_order(t_train, *best.popt)
    elif best.name == "Higuchi":
        y_fit = higuchi(t_train, *best.popt)
    else:
        y_fit = korsmeyer_peppas(t_train, *best.popt)

    res = y_train - y_fit

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(t_train, res, s=14, c="#2ca02c")
    plt.axhline(0, c="k", ls="--", alpha=0.7)
    plt.title("残差 vs 时间")
    plt.xlabel("时间 (h)")

    plt.subplot(122)
    plt.hist(res, bins=10, color="#2ca02c", alpha=0.7, edgecolor="white")
    plt.title("残差分布")
    plt.tight_layout()
    plt.savefig(OUT / "residuals.png", dpi=220, bbox_inches="tight")
    plt.close()

print("全部完成，输出在文件夹")
