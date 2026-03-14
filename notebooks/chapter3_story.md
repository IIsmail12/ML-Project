# 📖 Chapter 3: PLC Data Exploration
*Steel coil production line — data engineering story*

---

## 📋 Overview

We loaded a single CSV containing PLC sensor readings from a steel coil production
line. The dataset covers the same production period as the defect data (March–May 2019)
and will be merged with defect labels in Chapter 4.

| Metric | Value |
|--------|-------|
| **Total rows** | 299,447 |
| **Unique coils** | 1,261 |
| **Sensors** | 60 numeric channels |
| **Key columns** | `COIL`, `MT` (meter position), `DATE` |
| **Avg rows per coil** | 237.4 (~7m intervals → ~1,660m avg coil length) |
| **Min / Max rows per coil** | 2 / 1,107 |

### Sensor Groups
The 60 sensors span the full production line across 9 physical sections:

| Group | Sensors | Description |
|-------|---------|-------------|
| `speed` | 2 | Line and finishing speed |
| `furnace_temp` | 6 | Zone temperatures Z1–Z6 |
| `furnace_pyro` | 3 | Pyrometers pre-chamber and zones |
| `furnace_laser` | 9 | Strip position lasers along furnace |
| `furnace_air` | 18 | Air/CH4 ratio per zone (paired _1/_2) |
| `furnace_gas` | 12 | Gas flow per zone (paired _1/_2) |
| `furnace_pressure` | 9 | Zone pressures and furnace draft |
| `furnace_misc` | 8 | Extractors, ventilation, oven speed |
| `cooling` | 6 | Cooling circuits and speed |
| `raffination_laser` | 9 | Strip measurement post-furnace |
| `raffination_water` | 2 | Water ramps and outlet temperature |
| `electrolytic` | 4 | Bath chemistry and current |
| `descaling_chemistry` | 6 | Acid concentrations and iron content |
| `squeezers` | 6 | Squeezer roll pressures |
| `pickling_process` | 1 | Pickling tension |
| `finishing` | 5 | Force, tension, bending, alignment |

---

## 🔍 Finding 1 — Short Coils

**19 coils (1.5%) have ≤10 PLC rows.**

Inspection of these coils showed normal sensor readings — they are genuine short
production runs, not data artifacts. However they provide very little signal for
ML training.

| Metric | Value |
|--------|-------|
| Count | 19 coils |
| % of dataset | 1.5% |
| Action | Flagged in `SHORT_COILS` |
| Recommendation | Exclude from ML training pending domain expert review |

**Flagged coil IDs:**
`389937, 393524, 394340, 394719, 394828, 395726, 396323, 400039, 400148, 403272,
404717, 406920, 406938, 407825, 408915, 418315, 418349, 418848, 422319`

---

## 🔍 Finding 2 — Missing Data is the Tail-Out Effect

**57 of 60 sensor columns have missing values, falling into 3 clean patterns.**

| Pattern | Missing % | Columns | Sensor Groups |
|---------|-----------|---------|---------------|
| **A — Finishing** | 9.51% | 6 | `finishing`, `speed` |
| **B — Raffination** | 5.26% | 30 | `raffination_laser`, `furnace_pressure`, `cooling`, `raffination_water`, `furnace_misc` |
| **C — Pickling** | 5.23% | 21 | `squeezers`, `pickling_process`, `electrolytic`, `descaling_chemistry` |

### Root Cause: Tail-Out Effect
Missing data is **not random and not specific to certain coil types** — it is
concentrated at the **end of coils**, where the strip physically exits each
processing section before the coil is fully wound.

| Pattern | Missing mean position | Present mean position | Difference |
|---------|----------------------|----------------------|------------|
| Finishing | 1,529m | 960m | +569m |
| Raffination | 1,451m | 990m | +461m |
| Pickling | 1,411m | 992m | +419m |

This was visually confirmed across multiple coils — data goes dark at the end
of each coil, not randomly throughout.

### Actions Taken
- **No imputation** — missing = sensor offline = physical meaning
- **3 binary indicator columns added** to capture tail-out zone as a feature:

| Column | Rows Flagged | % of Data |
|--------|-------------|-----------|
| `IS_TAIL_FINISHING` | 28,480 | 9.51% |
| `IS_TAIL_RAFFINATION` | 15,734 | 5.26% |
| `IS_TAIL_PICKLING` | 15,647 | 5.23% |

> 💡 XGBoost handles NaN natively — keep as-is for tree models.
> For linear models: impute with 0 and use indicator columns.

---

## 🔍 Finding 3 — AIR_CH4 Negative Values

**6 air/CH4 ratio sensors recorded small negative values in <0.63% of rows.**

| Sensor | Negative rows | Negative % | Min value |
|--------|--------------|------------|-----------|
| `AIR_CH4_1` | 76 | 0.03% | -155.81 |
| `AIR_CH4_2` | 1,110 | 0.37% | -209.25 |
| `AIR_CH4_3` | 1,892 | 0.63% | -243.83 |
| `AIR_CH4_4` | 569 | 0.19% | -18.87 |
| `AIR_CH4_5` | 113 | 0.04% | -0.37 |
| `AIR_CH4_6` | 209 | 0.07% | -237.55 |

**Root cause:** Sensor noise at burner shutdown boundary — when furnace burners
wind down at the tail of a coil, the air/CH4 ratio oscillates slightly below zero.
Negatives never occur simultaneously across all sensors (0 rows where all are negative),
confirming this is localised noise rather than a systematic fault.

**Action taken:** Clipped all `AIR_CH4_*` columns to 0 using `.clip(lower=0)`.

---

## 🔍 Finding 4 — No Stuck or Constant Sensors

All 60 sensors are actively varying across the dataset.

| Check | Result |
|-------|--------|
| Completely constant sensors (1 unique value) | ✅ None |
| Almost constant sensors (≤5 unique values) | ✅ None |

No sensors need to be dropped for lack of variance.

---

## 🔍 Finding 5 — Correlations (Noted for ML Team)

128 highly correlated sensor pairs (|r| ≥ 0.90) were identified, explained by
3 physical patterns:

| Pattern | Example | r | Explanation |
|---------|---------|---|-------------|
| Paired zone sensors | `AIR_Z1_1` vs `AIR_Z1_2` | 0.997 | Left/right or top/bottom measurement in same zone — **not duplicates** |
| Speed-coupled sensors | `LS_OVEN` vs `LASER_FRN_4–7` | 0.997 | Line speed drives both oven and laser readings |
| Furnace zone temperatures | `TEMP_Z1` vs `TEMP_Z6` | 0.912 | Zones run at similar target temperatures |

### Key Finding: Paired Sensors are NOT Duplicates
Initially `AIR_ZN_1` / `AIR_ZN_2` and `GAS_ZN_1` / `GAS_ZN_2` appeared to be
duplicate sensors (r=1.0). Row-level comparison showed **0% identical rows** with
mean differences up to 112 units — confirming they are independent physical sensors
(e.g. left/right burner in the same zone).

**Action:** All 60 sensors kept unchanged.
**Handed to ML team** for feature selection and dimensionality reduction post-merge.

---

## 🔍 Finding 6 — Process Stability

**The production process is very stable overall.**

| Metric | Value |
|--------|-------|
| Median within-coil CV | 0.039 |
| Mean within-coil CV | 0.065 |
| Coils with CV > 1.0 | 2 (coils 384920, 424012) |
| Coils with CV > 0.5 | 10 |

Most variable sensors: `PRES_ZONA1/2/3_MEASURED`, `FRN_PHF_PRESSURE`,
`FRN_FURNACE_PRESSURE` — furnace pressure fluctuates more than temperature,
which is physically expected.

**2 unstable coils flagged:** `384920` (CV=3.55), `424012` (CV=1.5+)

---

## 📊 All Flagged Coils Summary

| Flag | Count | % of Coils | Action |
|------|-------|------------|--------|
| Short coils (≤10 rows) | 19 | 1.5% | Exclude from ML training |
| Unstable coils (CV > 1.0) | 2 | 0.2% | Flag for domain review |
| **Total unique flagged** | **21** | **1.7%** | Stored in `ALL_FLAGGED_COILS` |

---

## 📦 Output: Cleaned PLC Dataset

| Property | Value |
|----------|-------|
| Shape | 299,447 rows × 63 columns (60 sensors + 3 tail indicators) |
| Flagged coils | 21 stored in `ALL_FLAGGED_COILS` |
| Sensors modified | `AIR_CH4_1–6` clipped to 0 |
| New columns added | `IS_TAIL_FINISHING`, `IS_TAIL_RAFFINATION`, `IS_TAIL_PICKLING` |
| Sensors dropped | None |

---

## ❓ Questions for Domain Expert

| # | Question |
|---|----------|
| 1 | What do `AIR_ZN_1` vs `AIR_ZN_2` physically represent? (left/right? top/bottom?) |
| 2 | Are short coils (≤10 rows) test runs or production rejects? |
| 3 | Are coils 384920 and 424012 known production incidents? |
| 4 | What units are `TIRO_FORNO` and `TIRO_DECAP`? (tension in N? kN?) |
| 5 | Is `FILL_ALL` an alignment measurement or a combined flag? |

---

## ⏭️ Next Step: Chapter 4 — Merge PLC + Defect Data

- Join on `COIL` + meter position (`MT` within `MT_FROM`–`MT_TO`)
- 1,261 PLC coils × 534 defect coils → check overlap
- Exclude `ALL_FLAGGED_COILS` from final ML training set
- Output: one row per PLC reading with defect labels attached

---

*Chapter 3 completed — March 11, 2026*
*Status: ✅ Ready for merge*
