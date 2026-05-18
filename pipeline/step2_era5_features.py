"""
step2_era5_features.py
=======================
Шаг 2 из 3: Извлечение ERA5 признаков из NetCDF файла.

Вход:
  data/era5.nc                                      — ERA5 реанализ (скачать см. ниже)
  data/train_dataset.csv                            — для alignment по времени
  data/valid_features.csv                           — то же

Выход:
  data/extra_features_train.csv  — ERA5 признаки для train
  data/extra_features_valid.csv  — ERA5 признаки для valid

Как скачать ERA5 NC файл:
  1. Зарегистрируйся на https://cds.climate.copernicus.eu/
  2. Установи: pip install cdsapi
  3. Настрой ~/.cdsapirc (ключ API)
  4. Запусти step2_download_era5.py (отдельный скрипт)

  ИЛИ: если era5.nc уже есть — просто запускай этот скрипт.

Запуск:
  python3 step2_era5_features.py

Время: ~2-3 минуты
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── Пути ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"

ERA5_NC    = DATA_DIR / "era5.nc"          # ERA5 NetCDF файл
TRAIN_CSV  = DATA_DIR / "train_dataset.csv"
VALID_CSV  = DATA_DIR / "valid_features.csv"
OUT_TRAIN  = DATA_DIR / "extra_features_train.csv"
OUT_VALID  = DATA_DIR / "extra_features_valid.csv"

DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
R_AIR        = 287.05   # газовая постоянная воздуха (Дж/кг/К)
EPS          = 1e-6


def load_era5(nc_path: Path) -> pd.DataFrame:
    """
    Читает ERA5 NC файл и возвращает DataFrame с признаками.

    Важные баги при работе с ERA5 через xarray:

    БАГ 1: ds.interp() на single-point файле (только одна lat/lon точка)
    делает ЭКСТРАПОЛЯЦИЮ и возвращает 100% NaN.
    РЕШЕНИЕ: использовать ds.sel() с реальными координатами из файла.

    БАГ 2: pd.to_datetime(ds.valid_time.values) возвращает datetime64[ns],
    но pandas CSV читается как datetime64[us]. Merge тихо провалится (всё NaN).
    РЕШЕНИЕ: явно кастить к "datetime64[us]" перед созданием DataFrame.
    """
    print(f"  Читаем ERA5: {nc_path.name}")
    ds = xr.open_dataset(nc_path)

    # БАГ 1 FIX: берём реальные координаты из файла
    lat_avail = float(ds.latitude.values[0])
    lon_avail = float(ds.longitude.values[0])
    print(f"  ERA5 точка: lat={lat_avail}, lon={lon_avail}")
    ds_pt = ds.sel(latitude=lat_avail, longitude=lon_avail)

    # Извлекаем переменные
    u100 = ds_pt["u100"].values.ravel().astype(float)   # U-компонента 100м (м/с)
    v100 = ds_pt["v100"].values.ravel().astype(float)   # V-компонента 100м (м/с)
    u10  = ds_pt["u10"].values.ravel().astype(float)    # U-компонента 10м
    v10  = ds_pt["v10"].values.ravel().astype(float)    # V-компонента 10м
    t2m  = ds_pt["t2m"].values.ravel().astype(float)    # Температура 2м (К)
    sp   = ds_pt["sp"].values.ravel().astype(float)     # Давление поверхности (Па)
    blh  = ds_pt["blh"].values.ravel().astype(float)    # Высота пограничного слоя (м)

    # БАГ 2 FIX: явный каст datetime к us
    times = pd.to_datetime(ds_pt.valid_time.values.astype("datetime64[us]"))

    # Вычисляем признаки
    ws100 = np.sqrt(u100 ** 2 + v100 ** 2)   # скорость на 100м (хаб)
    ws10  = np.sqrt(u10  ** 2 + v10  ** 2)   # скорость на 10м

    era5_df = pd.DataFrame({
        "datetime"   : times,
        "era5_ws100" : ws100,                              # скорость ветра на 100м (хаб)
        "era5_blh"   : blh,                               # высота пограничного слоя
    })

    # Open-Meteo данные (ws200, t180) добавляются в step2_openmeteo.py отдельно.
    # Здесь только ERA5 переменные.

    print(f"  ERA5 строк: {len(era5_df)}")
    print(f"  Период: {times[0]} — {times[-1]}")
    print(f"  era5_ws100: min={ws100.min():.2f} max={ws100.max():.2f} м/с")
    print(f"  era5_blh:   min={blh.min():.0f} max={blh.max():.0f} м")

    return era5_df


def align_to_dataset(era5_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Выравнивает ERA5 признаки по временной шкале датасета.
    Merge по datetime (час в час).
    """
    df_raw = pd.read_csv(csv_path)
    df_raw["datetime"] = pd.to_datetime(df_raw[DATETIME_COL]).dt.tz_localize(None)
    dt_series = pd.DataFrame({"datetime": df_raw["datetime"].values})

    # Убеждаемся что ERA5 datetime тоже без timezone
    era5_sub = era5_df.copy()
    era5_sub["datetime"] = era5_sub["datetime"].dt.tz_localize(None) \
        if era5_sub["datetime"].dt.tz else era5_sub["datetime"]

    merged = dt_series.merge(era5_sub, on="datetime", how="left")

    nan_ws100 = merged["era5_ws100"].isna().mean() * 100
    nan_blh   = merged["era5_blh"].isna().mean()   * 100
    print(f"    NaN era5_ws100: {nan_ws100:.1f}%  era5_blh: {nan_blh:.1f}%")

    return merged


def add_openmeteo_columns(out_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Добавляет wind_speed_200m и temperature_180m из Open-Meteo.
    Эти данные качаются отдельно, но для удобства
    они уже включены в extra_features_train/valid.csv оригинального пайплайна.

    Если у тебя их нет — заполняем через power-law из доступных высот.
    """
    df_raw = pd.read_csv(csv_path)

    # Заполняем wind_speed_200m через power-law от 120m и 180m
    v180 = df_raw["wind_speed_180m"].clip(lower=EPS).values
    v120 = df_raw["wind_speed_120m"].clip(lower=EPS).values
    alpha = np.clip(np.log(np.maximum(v180, EPS) / np.maximum(v120, EPS)) / np.log(180/120), 0, 2)
    ws200 = v180 * (200 / 180) ** alpha

    # temperature_180m: линейная экстраполяция от 80m и 120m
    t80  = df_raw["temperature_80m"].values
    t120 = df_raw["temperature_120m"].values
    gradient = (t120 - t80) / (120 - 80)
    t180 = t120 + gradient * (180 - 120)

    out_df["wind_speed_200m"]  = ws200
    out_df["temperature_180m"] = t180
    return out_df


if __name__ == "__main__":
    print("=" * 60)
    print("Шаг 2: Извлечение ERA5 признаков")
    print("=" * 60)

    if not ERA5_NC.exists():
        print(f"\n❌ ERA5 файл не найден: {ERA5_NC}")
        print("\nКак скачать ERA5:")
        print("  1. pip install cdsapi")
        print("  2. Зарегистрируйся на https://cds.climate.copernicus.eu/")
        print("  3. Создай ~/.cdsapirc с API ключом")
        print("  4. Запусти: python3 step2_download_era5.py")
        print("\nИли: скопируй готовый era5.nc в папку data/")
        print("Файл: era5_data/era5_extracted/reanalysis-era5-single-levels-timeseries-sfcipx2cksg.nc")
        exit(1)

    # Загружаем ERA5
    era5_df = load_era5(ERA5_NC)

    # Выравниваем по train и valid
    for csv_path, out_path, name in [
        (TRAIN_CSV, OUT_TRAIN, "train"),
        (VALID_CSV, OUT_VALID, "valid"),
    ]:
        print(f"\n  Обрабатываем {name}...")
        out_df = align_to_dataset(era5_df, csv_path)
        out_df = add_openmeteo_columns(out_df, csv_path)
        out_df.to_csv(out_path, index=False)
        print(f"  ✅ Сохранено: {out_path.name}  ({len(out_df)} строк)")

    print("\n✅ Готово!")
    print(f"  {OUT_TRAIN.name}")
    print(f"  {OUT_VALID.name}")
