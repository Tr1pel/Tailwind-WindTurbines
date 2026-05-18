"""
reproduce_exp36.py
==================
Воспроизведение лучшего результата — exp36 (CatBoost + LightGBM blend).

Proxy LB: 8.6449 (лучший из всех экспериментов)
Ожидаемый LB: ~7.70–7.72 (если proxy→LB transfer ≈ 50%)

Запуск:
    python3 reproduce_exp36.py

Время: ~3-4 часа (обучение 8 моделей × 4 seeds = 32 модели)

Результаты:
    submissions/submission_36_ph45_mlblend55.csv  ← приоритет 1
    submissions/submission_36_ph40_mlblend60.csv  ← приоритет 2
    submissions/submission_36_ph50_mlblend50.csv  ← приоритет 3
    submissions/submission_36_lgbm_ml.csv         ← только LGBM
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import lightgbm as lgb

# ── ПУТИ ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
TRAIN_ENRICH = DATA_DIR / "train_dataset_enriched_gust_clip_all180.csv"
VALID_ENRICH = DATA_DIR / "valid_features_enriched_gust_clip_all180.csv"
TRAIN_ORIG   = DATA_DIR / "train_dataset.csv"
VALID_ORIG   = DATA_DIR / "valid_features.csv"
PH85_PATH    = DATA_DIR / "ph85.csv"
EXTRA_TRAIN  = DATA_DIR / "extra_features_train.csv"
EXTRA_VALID  = DATA_DIR / "extra_features_valid.csv"
OUTPUT_DIR   = ROOT / "submissions"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── КОНСТАНТЫ ─────────────────────────────────────────────────────────────────
DATETIME_COL    = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET          = "Выработка. Результирующий расчет"
MAX_POWER       = 90.09   # суммарная мощность фермы (МВт)
EPS             = 1e-6
SEEDS           = [42, 123, 456, 789]
SEASONAL_MONTHS = {11, 12, 1, 2, 3, 4}  # ноябрь–апрель

# ── Гиперпараметры моделей ─────────────────────────────────────────────────────
CB_PARAMS = dict(
    loss_function="MAE", eval_metric="MAE",
    iterations=5000, learning_rate=0.02,
    depth=8, l2_leaf_reg=12,
    allow_writing_files=False, verbose=0
)

LGBM_PARAMS = dict(
    objective="mae", metric="mae",
    n_estimators=5000, learning_rate=0.02,
    num_leaves=127, max_depth=-1,
    min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1,
)

# Q1-2025 = holdout для early stopping
HOLDOUT_TRAIN_END = "2024-12-31 23:59"
HOLDOUT_VAL_START = "2025-01-01"
HOLDOUT_VAL_END   = "2025-03-31 23:59"


# ── ЗАГРУЗКА ДАННЫХ ───────────────────────────────────────────────────────────
print("Загрузка данных...")
train_raw = pd.read_csv(TRAIN_ORIG)
valid_raw = pd.read_csv(VALID_ORIG)
train_enr = pd.read_csv(TRAIN_ENRICH)
valid_enr = pd.read_csv(VALID_ENRICH)
extra_tr  = pd.read_csv(EXTRA_TRAIN)
extra_va  = pd.read_csv(EXTRA_VALID)

train_raw["datetime"] = pd.to_datetime(train_raw[DATETIME_COL])
train_raw = train_raw.sort_values("datetime").reset_index(drop=True)
train_enr["datetime"]  = train_raw["datetime"].values
train_enr["month_num"] = train_enr["datetime"].dt.month
valid_raw["datetime"]  = pd.to_datetime(valid_raw[DATETIME_COL])
valid_enr["datetime"]  = valid_raw["datetime"].values
print(f"  Train: {len(train_raw)} строк  Valid: {len(valid_raw)} строк")


# ── ДОБАВЛЕНИЕ ERA5 ПРИЗНАКОВ (exp32) ─────────────────────────────────────────
print("Добавление ERA5 признаков...")

# Копируем 4 базовых ERA5 признака из extra_features
EXP32_EXTRA_COLS = ["era5_ws100", "era5_blh", "wind_speed_200m", "temperature_180m"]
for col in EXP32_EXTRA_COLS:
    if col in extra_tr.columns:
        train_enr[col] = extra_tr[col].values
        valid_enr[col] = extra_va[col].values

# BLH: заполняем NaN медианой по месяцу
if train_enr["era5_blh"].isna().any():
    med = train_enr.groupby(train_enr["month_num"])["era5_blh"].transform("median")
    train_enr["era5_blh"] = train_enr["era5_blh"].fillna(med)

# wind_speed_200m: заполняем power-law экстраполяцией
def fill_ws200(df_enr, df_raw):
    if df_enr["wind_speed_200m"].isna().any():
        v180 = df_raw["wind_speed_180m"].clip(lower=EPS).values
        v120 = df_raw["wind_speed_120m"].clip(lower=EPS).values
        alpha = np.clip(np.log(np.maximum(v180, EPS) / np.maximum(v120, EPS)) / np.log(180/120), 0, 2)
        extrap = v180 * (200/180)**alpha
        df_enr["wind_speed_200m"] = df_enr["wind_speed_200m"].fillna(
            pd.Series(extrap, index=df_enr.index))
    return df_enr

train_enr = fill_ws200(train_enr, train_raw)
valid_enr = fill_ws200(valid_enr, valid_raw)

# Производные ERA5 признаки
for df_enr, df_raw in [(train_enr, train_raw), (valid_enr, valid_raw)]:
    df_enr["era5_ws100_v3"]    = df_enr["era5_ws100"].clip(lower=0) ** 3
    v80 = df_raw["wind_speed_80m"].clip(lower=EPS).values
    df_enr["era5_nwp_ratio"]   = (df_enr["era5_ws100"] / np.maximum(v80, EPS)).clip(0.3, 3.0)
    df_enr["era5_blh_norm"]    = df_enr["era5_blh"] / 80.0
    df_enr["era5_stable"]      = (df_enr["era5_blh"] < 400).astype(float)
    df_enr["era5_blh_log"]     = np.log1p(df_enr["era5_blh"])
    v180 = df_raw["wind_speed_180m"].clip(lower=EPS).values
    df_enr["ws_200_180_shear"] = df_enr["wind_speed_200m"] - v180
    df_enr["ws200_v3"]         = df_enr["wind_speed_200m"].clip(lower=0) ** 3
    alpha_200 = np.clip(
        np.log(np.maximum(df_enr["wind_speed_200m"].values, EPS) / np.maximum(v180, EPS))
        / np.log(200/180), -2, 2)
    df_enr["alpha_180_200"]    = alpha_200

# Список признаков для обучения (301 признак)
EXCLUDE = {TARGET, "datetime", DATETIME_COL, "month_num"}
FCOLS = [c for c in train_enr.columns if c not in EXCLUDE and train_enr[c].dtype != object]
print(f"  Итого признаков: {len(FCOLS)}")


# ── МАСКИ ДЛЯ ОБУЧЕНИЯ ────────────────────────────────────────────────────────
dt = train_enr["datetime"]
SEASONAL_MASK = train_enr["month_num"].isin(SEASONAL_MONTHS)

# Holdout для early stopping
mask_tr_es = (dt <= HOLDOUT_TRAIN_END) & SEASONAL_MASK
mask_va_es = (dt >= HOLDOUT_VAL_START) & (dt <= HOLDOUT_VAL_END)

X_tr_es = train_enr.loc[mask_tr_es, FCOLS].fillna(-999)
y_tr_es  = train_enr.loc[mask_tr_es, TARGET]
X_va_es  = train_enr.loc[mask_va_es, FCOLS].fillna(-999)
y_va_es  = train_enr.loc[mask_va_es, TARGET]

# Полный seasonal train для финальных моделей
X_full = train_enr.loc[SEASONAL_MASK, FCOLS].fillna(-999)
y_full = train_enr.loc[SEASONAL_MASK, TARGET]

# Valid (предсказание)
X_test = valid_enr[FCOLS].fillna(-999)


# ── CATBOOST: EARLY STOPPING → ФИНАЛЬНЫЕ МОДЕЛИ ───────────────────────────────
print("\nCatBoost — early stopping на Q1-2025...")
cb_iters = []
for seed in SEEDS:
    m = CatBoostRegressor(**{**CB_PARAMS, "random_seed": seed})
    m.fit(X_tr_es, y_tr_es, eval_set=(X_va_es, y_va_es),
          use_best_model=True, early_stopping_rounds=200)
    p = np.clip(m.predict(X_va_es), 0, MAX_POWER)
    mae = mean_absolute_error(y_va_es, p)
    print(f"  seed={seed}: iter={m.best_iteration_}  MAE(Q1-25)={mae:.4f}")
    cb_iters.append(m.best_iteration_)

cb_final_iter = int(np.median(cb_iters))
print(f"CatBoost финальные модели (iter={cb_final_iter})...")
cb_models = []
for seed in SEEDS:
    m = CatBoostRegressor(**{**CB_PARAMS, "random_seed": seed, "iterations": cb_final_iter})
    m.fit(X_full, y_full)
    cb_models.append(m)

p_cb = np.clip(np.mean([m.predict(X_test) for m in cb_models], axis=0), 0, MAX_POWER)
print(f"  CB mean={p_cb.mean():.4f}")


# ── LIGHTGBM: EARLY STOPPING → ФИНАЛЬНЫЕ МОДЕЛИ ──────────────────────────────
print("\nLightGBM — early stopping на Q1-2025...")
lgbm_iters = []
for seed in SEEDS:
    params = {**LGBM_PARAMS, "random_state": seed}
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr_es, y_tr_es,
          eval_set=[(X_va_es, y_va_es)],
          callbacks=[lgb.early_stopping(200, verbose=False),
                     lgb.log_evaluation(period=-1)])
    p = np.clip(m.predict(X_va_es), 0, MAX_POWER)
    mae = mean_absolute_error(y_va_es, p)
    print(f"  seed={seed}: iter={m.best_iteration_}  MAE(Q1-25)={mae:.4f}")
    lgbm_iters.append(m.best_iteration_)

lgbm_final_iter = int(np.median(lgbm_iters))
print(f"LightGBM финальные модели (iter={lgbm_final_iter})...")
lgbm_models = []
for seed in SEEDS:
    params = {**LGBM_PARAMS, "random_state": seed, "n_estimators": lgbm_final_iter}
    m = lgb.LGBMRegressor(**params)
    m.fit(X_full, y_full)
    lgbm_models.append(m)

p_lgbm = np.clip(np.mean([m.predict(X_test) for m in lgbm_models], axis=0), 0, MAX_POWER)
print(f"  LGBM mean={p_lgbm.mean():.4f}")


# ── ML BLEND (50% CB + 50% LGBM) ─────────────────────────────────────────────
p_ml_blend = np.clip(0.5 * p_cb + 0.5 * p_lgbm, 0, MAX_POWER)
print(f"  ML blend mean={p_ml_blend.mean():.4f}")


# ── СОХРАНЕНИЕ SUBMISSIONS ────────────────────────────────────────────────────
print("\nСохранение submissions...")

pd.DataFrame({"prediction": p_cb}).to_csv(
    OUTPUT_DIR / "submission_36_cb_ml.csv", index=False)
print(f"  submission_36_cb_ml.csv  mean={p_cb.mean():.4f}")

pd.DataFrame({"prediction": p_lgbm}).to_csv(
    OUTPUT_DIR / "submission_36_lgbm_ml.csv", index=False)
print(f"  submission_36_lgbm_ml.csv  mean={p_lgbm.mean():.4f}")

pd.DataFrame({"prediction": p_ml_blend}).to_csv(
    OUTPUT_DIR / "submission_36_mlblend.csv", index=False)
print(f"  submission_36_mlblend.csv  mean={p_ml_blend.mean():.4f}")

# Бленды с ph85
if PH85_PATH.exists():
    ph85 = pd.read_csv(PH85_PATH).iloc[:, 0].values  # нет заголовка "prediction"
    configs = [
        ("submission_36_ph45_mlblend55.csv", 0.45, 0.55),  # аналог текущего лучшего
        ("submission_36_ph40_mlblend60.csv", 0.40, 0.60),
        ("submission_36_ph50_mlblend50.csv", 0.50, 0.50),
        ("submission_36_ph35_mlblend65.csv", 0.35, 0.65),
    ]
    for fname, w_ph, w_ml in configs:
        blend = np.clip(w_ph * ph85 + w_ml * p_ml_blend, 0, MAX_POWER)
        pd.DataFrame({"prediction": blend}).to_csv(OUTPUT_DIR / fname, index=False)
        print(f"  {fname}  mean={blend.mean():.4f}  (ph85={w_ph:.0%} ML={w_ml:.0%})")
else:
    print("  WARNING: ph85.csv не найден — только ML submissions")

print("\n" + "=" * 60)
print("ИТОГ exp36:")
print(f"  CatBoost mean:   {p_cb.mean():.4f}")
print(f"  LightGBM mean:   {p_lgbm.mean():.4f}")
print(f"  ML Blend mean:   {p_ml_blend.mean():.4f}")
print()
print("  Приоритет подачи на LB:")
print("    1. submission_36_ph45_mlblend55.csv")
print("    2. submission_36_ph40_mlblend60.csv")
print("    3. submission_36_ph50_mlblend50.csv")
print("    4. submission_36_lgbm_ml.csv")
print()
print("  Multi-Q1 CV proxy: 8.6449  (exp32 baseline: 8.7336)")
print("  Текущий LB:   7.744367")
print("  Цель:         < 7.73")
