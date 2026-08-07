"""Realistic synthetic pedestrian-count data (City of Melbourne schema).

Generates hourly counts per sensor with:
  - strong daily seasonality (morning/evening commute peaks)
  - weekly seasonality (weekends quieter; retail streets peak Sat)
  - annual seasonality + slow growth trend
  - public holidays (Australia/VIC via the `holidays` package)
  - occasional event spikes (NYE, Boxing Day, Melbourne Cup, random events)
  - overdispersion via negative-binomial noise
  - 3 of 8 sensors with deliberately short history (installed mid-series)
All deterministic (fixed seed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays as py_holidays

from .. import config as cfg

SENSOR_PROFILES = [
    # id, name, weekday peak mult (am, pm), weekend mult, base hourly mean
    (1,  "Bou292_T",  (2.6, 2.9), 0.75, 420.0),
    (2,  "Bou283_T",  (2.4, 2.7), 0.80, 380.0),
    (3,  "Swa295_T",  (2.8, 3.1), 0.70, 520.0),
    (4,  "Swa123_T",  (2.3, 2.6), 0.65, 310.0),
    (5,  "PriNW_T",   (1.9, 2.2), 0.60, 180.0),
    (30, "NewEliz_T", (2.5, 2.8), 0.70, 260.0),
    (31, "NewFla_T",  (2.2, 2.5), 0.72, 200.0),
    (32, "NewCol_T",  (2.4, 2.7), 0.68, 240.0),
]
SHORT_SENSOR_IDS = set(cfg.SYNTHETIC["short_ids"])


def _holiday_mask(dates: pd.DatetimeIndex) -> pd.Series:
    vic = py_holidays.AU(prov="VIC", years=range(dates.year.min(), dates.year.max() + 1))
    return pd.Series(pd.DatetimeIndex(dates).isin(vic), index=dates)


def make_synthetic_counts(all_sensors: bool = False) -> pd.DataFrame:
    s = cfg.SYNTHETIC
    rng = np.random.default_rng(cfg.SEED)
    start, end = pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
    hourly = pd.date_range(start, end, freq="h")
    short_starts = {loc: pd.Timestamp(t) for loc, t in zip(s["short_ids"], s["short_starts"])}

    special = {  # month-day -> multiplier (VIC-specific event calendar)
        ("01-01"): 2.6, ("12-31"): 2.8, ("12-26"): 1.7, ("11-07"): 1.5,  # NYD/NYE/Boxing/Melb Cup Tue
    }
    event_days = rng.choice(hourly.normalize().unique(), size=10, replace=False)

    if all_sensors:
        from .load import load_sensor_locations
        loc_df = load_sensor_locations()
        if not loc_df.empty:
            all_profiles = []
            for loc_id, row in loc_df.iterrows():
                loc_id = int(row.get("location_id", loc_id))
                name = str(row.get("sensor_name", f"Sensor {loc_id}"))
                base = 400.0 if loc_id in (1, 2, 3) else 250.0
                all_profiles.append((loc_id, name, (2.5, 2.8), 0.75, base))
        else:
            all_profiles = SENSOR_PROFILES
    else:
        all_profiles = SENSOR_PROFILES

    frames = []
    for loc_id, name, (ampm, pmpm), wend_mult, base in all_profiles:
        if loc_id in SHORT_SENSOR_IDS:
            t = hourly[hourly >= short_starts[loc_id]]
        else:
            t = hourly
        n = len(t)
        dow = t.dayofweek.to_numpy()          # Mon=0 .. Sun=6
        hour = t.hour.to_numpy()

        # daily shape: weekday commute peaks, night floor, weekend midday hump
        day_shape = np.ones(n)
        for h in range(24):
            m = hour == h
            if h <= 5:
                day_shape[m] = 0.10
            elif h == 8:
                day_shape[m] = ampm
            elif h == 17:
                day_shape[m] = pmpm
            elif 9 <= h <= 15:
                day_shape[m] = 0.75
            elif 18 <= h <= 21:
                day_shape[m] = 0.55
            else:
                day_shape[m] = 0.25
        # weekday/weekend
        is_wknd = dow >= 5
        wknd_floor, wknd_mid = 0.60, 0.95
        day_shape[is_wknd & ((hour < 7) | (hour >= 20))] = wknd_floor * 0.15
        day_shape[is_wknd & ((hour >= 9) & (hour <= 17))] = wknd_mid * 1.15 if loc_id in (1, 2) else wknd_mid
        day_shape[is_wknd & ~((hour >= 9) & (hour <= 17)) & (hour >= 7) & (hour < 20)] = wknd_floor

        # annual: peak in mid-summer (Jan), trough in winter (Jul); slow growth
        doy = t.dayofyear.to_numpy()
        annual = 1.0 + 0.22 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        growth = 1.0 + 0.02 * ((t - start).days / 365.25)

        # calendar multipliers
        hol = _holiday_mask(t)
        cal = np.ones(n)
        cal[hol.to_numpy()] = 0.30                       # public holidays: CBD dead
        for md, mult in special.items():
            cal[(t.strftime("%m-%d") == md)] = mult
        cal[np.isin(t.normalize().to_numpy(), event_days)] *= 1.9

        mu = base * day_shape * annual * growth * cal * wend_mult
        # negative-binomial overdispersion: variance = mu + 0.30*mu^2
        r = 1.0 / 0.30
        counts = rng.negative_binomial(r, r / (r + mu)).astype(int)

        # split into two directional streams (schema fidelity)
        d1 = rng.binomial(counts, 0.52)
        frames.append(pd.DataFrame({
            "id": np.arange(n, dtype=np.int64) + loc_id * 10_000_000,
            "location_id": loc_id,
            "sensing_date": t.normalize(),
            "hourday": t.hour,
            "direction_1": d1,
            "direction_2": counts - d1,
            "pedestriancount": counts,
            "sensor_name": name,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = df["sensing_date"] + pd.to_timedelta(df["hourday"], unit="h")
    out = pd.DataFrame({
        "location_id": df["location_id"],
        "datetime": df["datetime"],
        "count": df["pedestriancount"],
        "sensor_name": df["sensor_name"],
    })
    print(f"synthetic counts: {len(out):,} rows, {out.location_id.nunique()} sensors "
          f"({out.datetime.min()} .. {out.datetime.max()})")
    return out
