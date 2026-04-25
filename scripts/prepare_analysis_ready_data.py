#!/usr/bin/env python3
"""Prepare analysis-ready malaria datasets from existing CSV files.

This script does not read the source PDFs. It starts from the CSV files in
`data/processed/` and writes clean modelling inputs to `data/analysis_ready/`.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "analysis_ready"

DISTRICT_CSV = PROCESSED_DIR / "district_malaria_2000_2024_from_pdf.csv"
STATE_SITUATION_CSV = PROCESSED_DIR / "state_malaria_situation_2021_2025_from_pdf.csv"
STATE_EPI_CSV = PROCESSED_DIR / "state_epidemiological_2024_2025_from_pdf.csv"
POPULATION_CSV = RAW_DIR / "state_population_2011.csv"

SELECTED_REGIONS = {"Odisha", "Mizoram", "Tripura"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(str(value)))
    except ValueError:
        return default


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(str(value))
    except ValueError:
        return default


def rate_per_100k(numerator: float, population: float) -> float:
    if not population or math.isnan(population):
        return math.nan
    return numerator / population * 100_000


def clean_district_data(population_by_state: dict[str, int]) -> tuple[list[dict[str, object]], list[str]]:
    raw_rows = read_csv(DISTRICT_CSV)
    clean_rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    duplicate_count = 0

    for row in raw_rows:
        year = to_int(row["year"])
        state = row["state"].strip()
        district = row["district"].strip()
        total_cases = to_int(row["total_cases"])
        total_deaths = to_int(row["total_deaths"])
        key = (year, state, district)

        if key in seen:
            duplicate_count += 1
        seen.add(key)

        population = population_by_state.get(state)
        clean_rows.append(
            {
                "year": year,
                "state": state,
                "district": district,
                "total_cases": total_cases,
                "total_deaths": total_deaths,
                "population_2011": population if population is not None else "",
                "cases_per_100k": rate_per_100k(total_cases, population) if population else "",
                "deaths_per_100k": rate_per_100k(total_deaths, population) if population else "",
            }
        )

    notes = [
        f"district_rows={len(clean_rows)}",
        f"duplicate_year_state_district_rows={duplicate_count}",
        f"missing_population_rows={sum(1 for row in clean_rows if row['population_2011'] == '')}",
    ]
    return clean_rows, notes


def aggregate_state_year(
    district_rows: list[dict[str, object]],
    population_by_state: dict[str, int],
) -> tuple[list[dict[str, object]], list[str]]:
    grouped: dict[tuple[int, str], dict[str, object]] = {}
    districts_by_group: dict[tuple[int, str], set[str]] = defaultdict(set)

    for row in district_rows:
        key = (to_int(row["year"]), str(row["state"]))
        if key not in grouped:
            grouped[key] = {
                "year": key[0],
                "state": key[1],
                "total_cases": 0,
                "total_deaths": 0,
            }
        grouped[key]["total_cases"] = to_int(grouped[key]["total_cases"]) + to_int(row["total_cases"])
        grouped[key]["total_deaths"] = to_int(grouped[key]["total_deaths"]) + to_int(row["total_deaths"])
        districts_by_group[key].add(str(row["district"]))

    state_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        row = grouped[key]
        state = str(row["state"])
        population = population_by_state.get(state)
        total_cases = to_int(row["total_cases"])
        total_deaths = to_int(row["total_deaths"])

        state_rows.append(
            {
                "year": row["year"],
                "state": state,
                "total_cases": total_cases,
                "total_deaths": total_deaths,
                "district_count": len(districts_by_group[key]),
                "population_2011": population if population is not None else "",
                "cases_per_100k": rate_per_100k(total_cases, population) if population else "",
                "deaths_per_100k": rate_per_100k(total_deaths, population) if population else "",
                "selected_region": "yes" if state in SELECTED_REGIONS else "no",
            }
        )

    notes = [
        f"state_year_rows={len(state_rows)}",
        f"states_or_uts={len({row['state'] for row in state_rows})}",
        f"year_range={min(row['year'] for row in state_rows)}-{max(row['year'] for row in state_rows)}",
        f"missing_population_rows={sum(1 for row in state_rows if row['population_2011'] == '')}",
    ]
    return state_rows, notes


def make_selected_region_data(state_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in state_rows if row["state"] in SELECTED_REGIONS]


def rolling_mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    return sum(values) / len(values)


def make_aiml_features(state_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    by_state: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in state_rows:
        if row["population_2011"] == "":
            continue
        by_state[str(row["state"])].append(row)

    feature_rows: list[dict[str, object]] = []
    next_year_rates: list[float] = []

    for state, rows in by_state.items():
        rows = sorted(rows, key=lambda row: to_int(row["year"]))
        for index, row in enumerate(rows):
            if index < 3 or index >= len(rows) - 1:
                continue

            lag1 = rows[index - 1]
            lag2 = rows[index - 2]
            lag3 = rows[index - 3]
            next_row = rows[index + 1]

            current_cases = to_float(row["total_cases"])
            previous_cases = to_float(lag1["total_cases"])
            current_rate = to_float(row["cases_per_100k"])
            previous_rate = to_float(lag1["cases_per_100k"])
            next_rate = to_float(next_row["cases_per_100k"])

            cases_yoy_change = (
                (current_cases - previous_cases) / previous_cases if previous_cases > 0 else math.nan
            )
            rate_yoy_change = (
                (current_rate - previous_rate) / previous_rate if previous_rate > 0 else math.nan
            )

            feature_row = {
                "state": state,
                "year": row["year"],
                "population_2011": row["population_2011"],
                "district_count": row["district_count"],
                "total_cases": row["total_cases"],
                "cases_per_100k": row["cases_per_100k"],
                "total_cases_lag1": lag1["total_cases"],
                "total_cases_lag2": lag2["total_cases"],
                "total_cases_lag3": lag3["total_cases"],
                "total_deaths_lag1": lag1["total_deaths"],
                "cases_per_100k_lag1": lag1["cases_per_100k"],
                "cases_per_100k_lag2": lag2["cases_per_100k"],
                "cases_per_100k_lag3": lag3["cases_per_100k"],
                "deaths_per_100k_lag1": lag1["deaths_per_100k"],
                "cases_rate_roll3": rolling_mean(
                    [
                        to_float(lag1["cases_per_100k"]),
                        to_float(lag2["cases_per_100k"]),
                        to_float(lag3["cases_per_100k"]),
                    ]
                ),
                "cases_yoy_change": cases_yoy_change,
                "rate_yoy_change": rate_yoy_change,
                "next_year_cases": next_row["total_cases"],
                "next_year_rate": next_rate,
                "selected_region": row["selected_region"],
            }
            feature_rows.append(feature_row)
            next_year_rates.append(next_rate)

    threshold = sorted(next_year_rates)[int(0.75 * (len(next_year_rates) - 1))]
    for row in feature_rows:
        row["high_risk_next_year"] = 1 if to_float(row["next_year_rate"]) >= threshold else 0

    notes = [
        f"aiml_rows={len(feature_rows)}",
        f"high_risk_threshold_cases_per_100k={threshold:.6f}",
        f"high_risk_rows={sum(to_int(row['high_risk_next_year']) for row in feature_rows)}",
    ]
    return feature_rows, notes


def main() -> None:
    population_rows = read_csv(POPULATION_CSV)
    population_by_state = {
        row["state"].strip(): to_int(row["population_2011"])
        for row in population_rows
        if row.get("state") and row.get("population_2011")
    }

    district_rows, district_notes = clean_district_data(population_by_state)
    state_rows, state_notes = aggregate_state_year(district_rows, population_by_state)
    selected_rows = make_selected_region_data(state_rows)
    aiml_rows, aiml_notes = make_aiml_features(state_rows)

    write_csv(
        OUT_DIR / "district_malaria_clean.csv",
        district_rows,
        [
            "year",
            "state",
            "district",
            "total_cases",
            "total_deaths",
            "population_2011",
            "cases_per_100k",
            "deaths_per_100k",
        ],
    )
    write_csv(
        OUT_DIR / "state_year_malaria_clean.csv",
        state_rows,
        [
            "year",
            "state",
            "total_cases",
            "total_deaths",
            "district_count",
            "population_2011",
            "cases_per_100k",
            "deaths_per_100k",
            "selected_region",
        ],
    )
    write_csv(
        OUT_DIR / "selected_regions_state_year.csv",
        selected_rows,
        [
            "year",
            "state",
            "total_cases",
            "total_deaths",
            "district_count",
            "population_2011",
            "cases_per_100k",
            "deaths_per_100k",
            "selected_region",
        ],
    )
    write_csv(
        OUT_DIR / "aiml_state_year_features.csv",
        aiml_rows,
        [
            "state",
            "year",
            "population_2011",
            "district_count",
            "total_cases",
            "cases_per_100k",
            "total_cases_lag1",
            "total_cases_lag2",
            "total_cases_lag3",
            "total_deaths_lag1",
            "cases_per_100k_lag1",
            "cases_per_100k_lag2",
            "cases_per_100k_lag3",
            "deaths_per_100k_lag1",
            "cases_rate_roll3",
            "cases_yoy_change",
            "rate_yoy_change",
            "next_year_cases",
            "next_year_rate",
            "high_risk_next_year",
            "selected_region",
        ],
    )

    report = OUT_DIR / "data_preparation_report.txt"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("Analysis-ready malaria data preparation report\n")
        handle.write("==============================================\n\n")
        for section, notes in [
            ("district_malaria_clean.csv", district_notes),
            ("state_year_malaria_clean.csv", state_notes),
            ("selected_regions_state_year.csv", [f"selected_region_rows={len(selected_rows)}"]),
            ("aiml_state_year_features.csv", aiml_notes),
        ]:
            handle.write(section + "\n")
            for note in notes:
                handle.write(f"  {note}\n")
            handle.write("\n")

        handle.write("Source CSV files\n")
        handle.write(f"  {DISTRICT_CSV}\n")
        handle.write(f"  {STATE_SITUATION_CSV}\n")
        handle.write(f"  {STATE_EPI_CSV}\n")
        handle.write(f"  {POPULATION_CSV}\n")

    print(f"Wrote analysis-ready data to {OUT_DIR}")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
