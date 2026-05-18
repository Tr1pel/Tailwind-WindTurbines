"""
step1_preprocess.py
====================
Шаг 1 из 3: Создание обогащённого датасета из исходных данных.

Вход:
  data/train_dataset.csv       — исходный train (32434 строк, 21 признак)
  data/valid_features.csv      — исходный valid (2126 строк, 20 признаков)

Выход:
  data/train_dataset_enriched_gust_clip_all180.csv  — 289 признаков
  data/valid_features_enriched_gust_clip_all180.csv — 289 признаков

Запуск:
  python3 step1_preprocess.py

Время: ~5-10 минут
"""
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Пути ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"

INPUTS = {
    "train": DATA_DIR / "train_dataset.csv",
    "valid": DATA_DIR / "valid_features.csv",
}
OUTPUTS = {
    "train": DATA_DIR / "train_dataset_enriched_gust_clip_all180.csv",
    "valid": DATA_DIR / "valid_features_enriched_gust_clip_all180.csv",
}

# ── Константы ─────────────────────────────────────────────────────────────────
DATETIME_COL      = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET            = "Выработка. Результирующий расчет"
REPAIR_COL        = "Кол-во_ВЭУ_в_ремонте"
TOTAL_TURBINES    = 26
TURBINE_RATED_MW  = 3.465
ROTOR_RADIUS_M    = 66
ROTOR_AREA_M2     = math.pi * ROTOR_RADIUS_M ** 2
EPS               = 1e-6

TIME_CYCLIC_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos",
                    "dayofyear_sin", "dayofyear_cos"]

WIND_DIR_COLS = ["wind_direction_10m", "wind_direction_80m",
                 "wind_direction_120m", "wind_direction_180m"]

# Столбцы для rolling/lag признаков
ROLLING_COLS = [
    "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_gusts_10m",
    "rotor_cubic_speed_10_80_120", "rotor_speed_cubed_10_80_120", "wind_power_available_mw",
    "rotor_cubic_speed_10_80_120_180", "rotor_speed_cubed_10_80_120_180",
    "wind_power_available_mw_10_80_120_180", "air_density",
]


# ── Функции добавления признаков ─────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Циклические временные признаки (sin/cos)."""
    dt    = pd.to_datetime(df[DATETIME_COL])
    hour  = df["hour_of_day"].astype(float)
    month = df["month"].astype(float)
    doy   = dt.dt.dayofyear.astype(float)

    df["hour_sin"]      = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"]     = np.sin(2 * np.pi * month / 12)
    df["month_cos"]     = np.cos(2 * np.pi * month / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * doy   / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * doy   / 365)
    return df


def add_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos направления ветра.
    Исходное значение нормировано: direction_deg = value × 1000 (диапазон [0, 360]).
    """
    for col in WIND_DIR_COLS:
        angle_rad        = 2 * np.pi * df[col] / 0.36
        df[f"{col}_sin"] = np.sin(angle_rad)
        df[f"{col}_cos"] = np.cos(angle_rad)
    return df


def fill_wind180(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняем пропуски wind_speed_180m и wind_direction_180m через power-law.
    В train ~21% NaN (ранние 2022 данные не содержали 180м измерений).
    """
    # wind_speed_180m: power-law экстраполяция от 80m и 120m
    mask = df["wind_speed_180m"].isna()
    if mask.any():
        v80  = df.loc[mask, "wind_speed_80m"].clip(lower=EPS)
        v120 = df.loc[mask, "wind_speed_120m"].clip(lower=EPS)
        alpha = np.clip(np.log(v120 / v80) / np.log(120 / 80), 0, 2)
        df.loc[mask, "wind_speed_180m"] = v120 * (180 / 120) ** alpha

    # wind_direction_180m: заполняем значением 120m
    if "wind_direction_180m" in df.columns:
        df["wind_direction_180m"] = df["wind_direction_180m"].fillna(df["wind_direction_120m"])
    return df


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Физические признаки: плотность воздуха, мощность, сдвиг ветра, кубы."""
    active = TOTAL_TURBINES - df[REPAIR_COL]

    # Флаги пропусков (до заполнения)
    df["wind_speed_180m_missing"]     = df["wind_speed_180m"].isna().astype(int)
    df["wind_direction_180m_missing"] = df["wind_direction_180m"].isna().astype(int)

    # Турбины
    df["Кол-во_ВЭУ_в_работе"]  = active
    df["available_capacity_mw"] = active * TURBINE_RATED_MW
    df["availability_frac"]     = active / TOTAL_TURBINES

    # Плотность воздуха: ρ = P/(R·T), P в Па, T в К
    df["air_density"] = df["pressure_msl"] * 100 / (287.05 * (df["temperature_80m"] + 273.15))

    # Площадь ротора
    df["rotor_area_m2"]           = ROTOR_AREA_M2
    df["effective_rotor_area_m2"] = active * ROTOR_AREA_M2

    # Кубы скоростей
    for h in ["10m", "80m", "120m", "180m"]:
        df[f"wind_speed_{h}_cubed"] = df[f"wind_speed_{h}"] ** 3

    # Rotor equivalent wind speed cubed (REWS³) — без и с 180м
    df["rotor_speed_cubed_10_80_120"] = (
        df["wind_speed_10m_cubed"] + df["wind_speed_80m_cubed"] + df["wind_speed_120m_cubed"]
    ) / 3
    df["rotor_speed_cubed_10_80_120_180"] = (
        df["wind_speed_10m_cubed"] + df["wind_speed_80m_cubed"] +
        df["wind_speed_120m_cubed"] + df["wind_speed_180m_cubed"]
    ) / 4

    df["rotor_cubic_speed_10_80_120"]     = df["rotor_speed_cubed_10_80_120"]     ** (1/3)
    df["rotor_cubic_speed_10_80_120_180"] = df["rotor_speed_cubed_10_80_120_180"] ** (1/3)

    # Доступная мощность (МВт) = 0.5 × ρ × A × REWS³ / 1e6
    df["wind_power_available_mw"] = (
        0.5 * df["air_density"] * df["effective_rotor_area_m2"] *
        df["rotor_speed_cubed_10_80_120"] / 1e6
    )
    df["wind_power_available_mw_10_80_120_180"] = (
        0.5 * df["air_density"] * df["effective_rotor_area_m2"] *
        df["rotor_speed_cubed_10_80_120_180"] / 1e6
    )

    # Сдвиг ветра (wind shear)
    df["wind_shear_80_10"]  = df["wind_speed_80m"]  - df["wind_speed_10m"]
    df["wind_shear_120_80"] = df["wind_speed_120m"] - df["wind_speed_80m"]
    df["wind_shear_180_120"]= df["wind_speed_180m"] - df["wind_speed_120m"]

    # Отношения скоростей между высотами
    for upper, lower in [("80","10"),("120","10"),("180","10"),
                         ("120","80"),("180","80"),("180","120")]:
        u = df[f"wind_speed_{upper}m"]
        l = df[f"wind_speed_{lower}m"]
        df[f"wind_ratio_{upper}_{lower}"]     = np.where(l > 0.5, u / l, np.nan)
        df[f"log_wind_ratio_{upper}_{lower}"] = np.log1p(u) - np.log1p(l)

    # Порывы ветра
    df["gust_minus_10m"]  = df["wind_gusts_10m"] - df["wind_speed_10m"]
    df["gust_factor_10m"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + EPS)

    # Температурный градиент
    df["temp_gradient_120_80"] = df["temperature_120m"] - df["temperature_80m"]

    # Осадки
    df["precip_total"] = df["rain"] + df["showers"] + df["snowfall"]
    df["has_precip"]   = (df["precip_total"] > 0).astype(int)
    df["has_snow"]     = (df["snowfall"] > 0).astype(int)
    df["freezing_flag"]= (df["temperature_80m"] <= 0).astype(int)
    df["icing_proxy"]  = df["has_precip"] * df["freezing_flag"]

    # ρ × v³ (физическая мощность)
    df["rho_v80_cubed"]  = df["air_density"] * df["wind_speed_80m_cubed"]
    df["rho_v120_cubed"] = df["air_density"] * df["wind_speed_120m_cubed"]
    df["rho_v180_cubed"] = df["air_density"] * df["wind_speed_180m_cubed"]
    df["rho_rotor_v_cubed"] = df["air_density"] * df["rotor_speed_cubed_10_80_120"]
    df["rho_rotor_v_cubed_10_80_120_180"] = df["air_density"] * df["rotor_speed_cubed_10_80_120_180"]

    # Упрощённая кривая мощности (ramp function)
    ramp     = ((df["rotor_cubic_speed_10_80_120"]     - 3) / (12 - 3)).clip(0, 1) ** 3
    ramp_180 = ((df["rotor_cubic_speed_10_80_120_180"] - 3) / (12 - 3)).clip(0, 1) ** 3
    df["simple_power_curve_mw"]             = df["available_capacity_mw"] * ramp
    df["simple_power_curve_mw_10_80_120_180"] = df["available_capacity_mw"] * ramp_180

    # U/V компоненты (декомпозиция вектора ветра)
    for h in ["10m", "80m", "120m", "180m"]:
        spd  = df[f"wind_speed_{h}"]
        ssin = df[f"wind_direction_{h}_sin"]
        scos = df[f"wind_direction_{h}_cos"]
        df[f"wind_u_{h}"]       = spd * ssin
        df[f"wind_v_{h}"]       = spd * scos
        df[f"wind_u_cubed_{h}"] = (spd ** 3) * ssin
        df[f"wind_v_cubed_{h}"] = (spd ** 3) * scos

    # Взаимодействие REWS × направление
    df["rotor_cubed_dir80_sin"]  = df["rotor_speed_cubed_10_80_120"] * df["wind_direction_80m_sin"]
    df["rotor_cubed_dir80_cos"]  = df["rotor_speed_cubed_10_80_120"] * df["wind_direction_80m_cos"]
    df["wind_power_dir80_sin"]   = df["wind_power_available_mw"]     * df["wind_direction_80m_sin"]
    df["wind_power_dir80_cos"]   = df["wind_power_available_mw"]     * df["wind_direction_80m_cos"]
    df["rotor_cubed_180aware_dir80_sin"] = df["rotor_speed_cubed_10_80_120_180"] * df["wind_direction_80m_sin"]
    df["rotor_cubed_180aware_dir80_cos"] = df["rotor_speed_cubed_10_80_120_180"] * df["wind_direction_80m_cos"]
    df["wind_power_180aware_dir80_sin"]  = df["wind_power_available_mw_10_80_120_180"] * df["wind_direction_80m_sin"]
    df["wind_power_180aware_dir80_cos"]  = df["wind_power_available_mw_10_80_120_180"] * df["wind_direction_80m_cos"]

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std и lag признаки по временной шкале."""
    original_index = df.index
    df_sorted = df.sort_values(DATETIME_COL)
    frames = []

    for col in ROLLING_COLS:
        if col not in df_sorted.columns:
            continue
        feats = {}
        for w in [3, 6, 12, 24]:
            roll = df_sorted[col].rolling(w, min_periods=1)
            feats[f"{col}_roll{w}_mean"] = roll.mean()
            feats[f"{col}_roll{w}_std"]  = roll.std()
        for lag in [1, 2, 3]:
            lagged = df_sorted[col].shift(lag)
            feats[f"{col}_lag{lag}"]      = lagged
            feats[f"{col}_lead{lag}"]     = df_sorted[col].shift(-lag)
            feats[f"{col}_diff_lag{lag}"] = df_sorted[col] - lagged
        frames.append(pd.DataFrame(feats, index=df_sorted.index))

    result = pd.concat([df_sorted, *frames], axis=1)
    return result.loc[original_index]


def ordered_columns(df: pd.DataFrame) -> list:
    """Упорядочиваем столбцы: сначала временные, потом target, потом остальные."""
    excluded = {DATETIME_COL, "month", "hour_of_day", REPAIR_COL}
    remaining = [c for c in df.columns if c not in excluded]
    cols = [c for c in TIME_CYCLIC_COLS if c in remaining]
    if TARGET in remaining:
        cols.append(TARGET)
    cols.extend(c for c in remaining if c not in set(cols))
    return cols


def process_file(input_path: Path, output_path: Path) -> None:
    print(f"  Читаем: {input_path.name}")
    df = pd.read_csv(input_path)
    print(f"    Строк: {len(df)}, Признаков исходных: {len(df.columns)}")

    df = add_time_features(df)
    df = add_direction_features(df)
    df = fill_wind180(df)          # заполнение пропусков 180м до физических фич
    df = add_physical_features(df)
    df = add_rolling_features(df)
    df = df[ordered_columns(df)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"    Сохранено: {output_path.name}  ({len(df.columns)} признаков)")


if __name__ == "__main__":
    print("=" * 60)
    print("Шаг 1: Создание обогащённого датасета")
    print("=" * 60)

    for name, input_path in INPUTS.items():
        if not input_path.exists():
            print(f"  ERROR: Файл не найден: {input_path}")
            continue
        process_file(input_path, OUTPUTS[name])

    print("\nOK: Готово!")
    print(f"  {OUTPUTS['train'].name}")
    print(f"  {OUTPUTS['valid'].name}")
