# Wind Power Forecasting — Полное воспроизведение

---

## Быстрый старт

```bash
pip install catboost lightgbm scikit-learn pandas numpy xarray cdsapi

# Положи исходные данные в папку data/:
#   data/train_dataset.csv
#   data/valid_features.csv
#   data/ph85.csv
#   data/era5.nc  (или скачай через step2_download_era5.py)

python3 step1_preprocess.py      # ~10 мин
python3 step2_era5_features.py   # ~3 мин
python3 step3_train_predict.py   # ~3-4 часа

# Результат: submissions/submission_36_ph45_mlblend55.csv  ← подавать на LB
```

## Навигация по файлам

- [README.md](README.md) — краткое описание решения.
- [pipeline/step1_preprocess.py](pipeline/step1_preprocess.py) — предобработка и feature engineering.
- [pipeline/step2_download_era5.py](pipeline/step2_download_era5.py) — скачивание ERA5 через CDS API.
- [pipeline/step2_era5_features.py](pipeline/step2_era5_features.py) — извлечение ERA5-признаков.
- [pipeline/step3_train_predict.py](pipeline/step3_train_predict.py) — обучение моделей и сборка submission.
- [pipeline/data/train_dataset.csv](pipeline/data/train_dataset.csv) и [pipeline/data/valid_features.csv](pipeline/data/valid_features.csv) — исходные датасеты в репозитории.

---

## Структура папки

```
pipeline/
├── README.md                    ← этот файл
├── step1_preprocess.py          ← создаёт обогащённый датасет (289 признаков)
├── step2_download_era5.py       ← скачивает ERA5 через CDS API (~1.5 ГБ)
├── step2_era5_features.py       ← извлекает ERA5 признаки → extra_features CSV
├── step3_train_predict.py       ← обучает CB + LGBM → submissions
│
├── data/
│   ├── train_dataset.csv        ← ПОЛОЖИТЬ СЮДА (исходный, 32434 строк)
│   ├── valid_features.csv       ← ПОЛОЖИТЬ СЮДА (исходный, 2126 строк)
│   ├── ph85.csv                 ← ПОЛОЖИТЬ СЮДА (физическая модель)
│   ├── era5.nc                  ← ПОЛОЖИТЬ СЮДА или скачать step2_download_era5.py
│   │
│   │   ── создаются автоматически ──
│   ├── train_dataset_enriched_gust_clip_all180.csv   ← step1 output
│   ├── valid_features_enriched_gust_clip_all180.csv  ← step1 output
│   ├── extra_features_train.csv  ← step2 output
│   └── extra_features_valid.csv  ← step2 output
│
└── submissions/                 ← step3 output (все варианты)
    ├── submission_36_ph45_mlblend55.csv  ← ПРИОРИТЕТ 1 (аналог лучшего LB)
    ├── submission_36_ph40_mlblend60.csv  ← приоритет 2
    ├── submission_36_ph50_mlblend50.csv  ← приоритет 3
    └── submission_36_lgbm_ml.csv         ← только LGBM (без ph85)
```

---

## Шаг 1: Создание обогащённого датасета

**Скрипт**: [pipeline/step1_preprocess.py](pipeline/step1_preprocess.py)

Из 21 исходного признака создаём 289 признаков:

| Группа | Признаки |
|---|---|
| Временные (циклические) | sin/cos часа, месяца, дня года |
| Направление ветра | sin/cos для 10m, 80m, 120m, 180m |
| Физические | ρ (плотность), v³, REWS³, доступная мощность |
| Сдвиг ветра | wind_shear и wind_ratio между высотами |
| Осадки | флаги (has_precip, has_snow, freezing_flag, icing_proxy) |
| U/V компоненты | декомпозиция вектора ветра |
| Rolling/Lag | mean/std/lag(1,2,3)/lead(1,2,3) за 3h/6h/12h/24h |

**Заполнение пропусков** (21% NaN в wind_speed_180m train):
```python
# Power-law экстраполяция от 120m → 180m
alpha = log(v120/v80) / log(120/80)
v180  = v120 * (180/120)^alpha
```

---

## Шаг 2: ERA5 признаки

**ERA5** — ECMWF глобальный реанализ (historical ground truth).  
В отличие от NWP прогноза, ERA5 использует ассимиляцию наблюдений → точнее.

### Скачать ERA5 (если нет файла)

```bash
pip install cdsapi

# Создай ~/.cdsapirc:
# url: https://cds.climate.copernicus.eu/api/v2
# key: <UID>:<API_KEY>
# Регистрация: https://cds.climate.copernicus.eu/

python3 step2_download_era5.py   # ~30-60 мин, файл ~1.5 ГБ
```

**Скрипт**: [pipeline/step2_download_era5.py](pipeline/step2_download_era5.py)

### ERA5 признаки (выходные)

**Скрипт извлечения признаков**: [pipeline/step2_era5_features.py](pipeline/step2_era5_features.py)

| Признак | Описание |
|---|---|
| `era5_ws100` | Скорость ветра на 100м (высота хаба) = √(u100² + v100²) |
| `era5_blh`   | Высота пограничного слоя атмосферы (м) |
| `wind_speed_200m`  | Экстраполяция 180→200м через power-law |
| `temperature_180m` | Экстраполяция температуры до 180м |

### Производные ERA5 признаки (создаются в step3)

```python
era5_ws100_v3    = era5_ws100³               # куб скорости хаба
era5_nwp_ratio   = era5_ws100 / wind_speed_80m  # ERA5 vs NWP отношение
era5_blh_norm    = era5_blh / 80             # BLH нормированная к хабу
era5_stable      = (era5_blh < 400)          # флаг стабильной атмосферы
era5_blh_log     = log(1 + era5_blh)
ws_200_180_shear = wind_speed_200m - wind_speed_180m
ws200_v3         = wind_speed_200m³
alpha_180_200    = log(ws200/ws180) / log(200/180)  # степенной профиль
```

**Итого признаков для модели: 301** (289 из step1 + 12 ERA5)

---

## Шаг 3: Обучение и предсказание

**Скрипт**: [pipeline/step3_train_predict.py](pipeline/step3_train_predict.py)

### Модели

#### CatBoost
```python
loss_function="MAE", learning_rate=0.02, depth=8, l2_leaf_reg=12
4 seeds: [42, 123, 456, 789]
```

#### LightGBM
```python
objective="mae", learning_rate=0.02, num_leaves=127
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
4 seeds: [42, 123, 456, 789]
```

### Ключевая настройка: сезонная фильтрация

```python
SEASONAL_MONTHS = {11, 12, 1, 2, 3, 4}  # ноябрь–апрель
```

Valid = Q1 2026 (январь–март). Обучаем ТОЛЬКО на зимних месяцах.
Это убирает летние данные, которые мешают модели учиться на зимних паттернах.

### Процедура

```
1. Early stopping на Q1-2025 (holdout) → находим оптимальное число итераций
2. Финальная модель на полном seasonal train (все годы)
3. Предсказание на valid_features.csv
4. Бленд: w_ph85 × ph85 + w_ml × (0.5×CB + 0.5×LGBM)
```

### Multi-Q1 CV результаты (из нашего эксперимента)

| Модель | Q1-2023 | Q1-2024 | Q1-2025 | LB-proxy |
|---|---|---|---|---|
| CatBoost | 7.2125 | 8.8046 | 9.6223 | 8.7336 |
| LightGBM | 7.2618 | 8.8588 | **9.4520** | 8.6968 |
| **Blend** | **7.1993** | **8.7714** | **9.4379** | **8.6449** |

LightGBM значительно лучше на Q1-2025 (самый важный фолд, вес=0.40).

---

## Финальный бленд

```
submission = clip(w_ph × ph85 + w_ml × ML_blend, 0, 90.09)
ML_blend   = 0.5 × CatBoost + 0.5 × LightGBM
```

| Файл | ph85 | ML | mean (МВт) | Приоритет LB |
|---|---|---|---|---|
| submission_36_ph45_mlblend55.csv | 45% | 55% | 39.08 | **1** |
| submission_36_ph40_mlblend60.csv | 40% | 60% | 39.09 | 2 |
| submission_36_ph50_mlblend50.csv | 50% | 50% | 39.07 | 3 |
| submission_36_lgbm_ml.csv        | 0%  | 100% LGBM | 39.18 | 4 |

**ph85.csv** — это предсказания физической модели тиммейта (REWS физика):
```
P = 0.5 × ρ × π·R² × REWS³ × cp_eta
REWS = rotor equivalent wind speed (R=67м, H=80м хаб)
cp_eta = 0.40
```

---

## Почему не подходит случайный split?

Valid = Q1 2026 — это БУДУЩЕЕ. Модель должна предсказывать будущее по прошлому.
Случайный split создаёт утечку данных: модель видит данные "после" тестового периода.

**Правильная стратегия** (Multi-Q1 CV):
- Fold 1: train 2022 → test Q1-2023
- Fold 2: train 2022-2023 → test Q1-2024  
- Fold 3: train 2022-2024 → test Q1-2025
- Final: train 2022-2025 → predict Q1-2026

**Критерий принятия новых признаков**: все 3 фолда должны улучшиться.

---

## Известные баги при работе с ERA5

### БАГ 1: xarray interp на single-point файле

```python
# НЕПРАВИЛЬНО — даёт 100% NaN (экстраполяция):
ds_pt = ds.interp(latitude=46.83, longitude=38.72)

# ПРАВИЛЬНО:
lat_avail = float(ds.latitude.values[0])   # реальная lat из файла
lon_avail = float(ds.longitude.values[0])  # реальная lon из файла
ds_pt = ds.sel(latitude=lat_avail, longitude=lon_avail)
```

### БАГ 2: datetime64 precision mismatch

```python
# ERA5 через xarray → datetime64[ns]
# pandas CSV → datetime64[us]
# pandas merge между ns и us тихо возвращает 100% NaN

# НЕПРАВИЛЬНО:
times = pd.to_datetime(ds_pt.valid_time.values)  # ns precision

# ПРАВИЛЬНО:
times = pd.to_datetime(ds_pt.valid_time.values.astype("datetime64[us]"))
```

---

## История LB

| Конфигурация | LB MAE |
|---|---|
| 60% ph85 + 40% ML (baseline) | 7.789204 |
| 20% cv2 + 30% ph85 + 50% ML | **7.758868** |
| 45% ph85 + 55% ML (CB, exp32) | **7.744367** ← текущий лучший |
| 45% ph85 + 55% CB+LGBM (exp36) | *ожидаем ~7.70* |

---

*Последнее обновление: 2026-05-18*
