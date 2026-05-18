"""
step2_download_era5.py
=======================
Скачивает ERA5 реанализ через CDS API.

Требования:
  pip install cdsapi
  Файл ~/.cdsapirc с содержимым:
    url: https://cds.climate.copernicus.eu/api/v2
    key: <твой_uid>:<твой_api_key>

Регистрация: https://cds.climate.copernicus.eu/

Что качаем:
  ERA5 hourly single-levels, 2022-01-01 — 2026-03-31
  Переменные: u100, v100, u10, v10, t2m, blh, sp
  Точка: 46.75°N, 38.75°E (ближайший ERA5 узел к станции 46.83°N, 38.72°E)
  Разрешение: 0.25°

Время загрузки: ~30-60 минут (файл ~1.5 ГБ)
"""
import cdsapi
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "era5.nc"

if OUT_FILE.exists():
    print(f"✅ ERA5 файл уже существует: {OUT_FILE}")
    print(f"   Размер: {OUT_FILE.stat().st_size / 1024**3:.2f} ГБ")
    exit(0)

c = cdsapi.Client()

print("Скачиваем ERA5...")
print(f"Выходной файл: {OUT_FILE}")

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format":       "netcdf",
        "variable": [
            "100m_u_component_of_wind",    # u100
            "100m_v_component_of_wind",    # v100
            "10m_u_component_of_wind",     # u10
            "10m_v_component_of_wind",     # v10
            "2m_temperature",              # t2m
            "boundary_layer_height",       # blh
            "surface_pressure",            # sp
        ],
        "year":  ["2022", "2023", "2024", "2025", "2026"],
        "month": ["01","02","03","04","05","06","07","08","09","10","11","12"],
        "day":   [f"{d:02d}" for d in range(1, 32)],
        "time":  [f"{h:02d}:00" for h in range(24)],
        # Bounding box вокруг точки (46.75°N, 38.75°E) с запасом 0.5°
        "area":  [47.25, 38.25, 46.25, 39.25],  # N/W/S/E
    },
    str(OUT_FILE),
)

print(f"\n✅ ERA5 скачан: {OUT_FILE}")
print(f"   Размер: {OUT_FILE.stat().st_size / 1024**3:.2f} ГБ")
print("\nТеперь запусти: python3 step2_era5_features.py")
