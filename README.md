# Malaria Risk Modelling Project

This folder contains the malaria datasets and notebooks for geospatial analysis, mechanistic modelling, time-series forecasting, and AI/ML risk prediction.

## Structure

- `data/processed/`: CSV files extracted from the PDFs.
- `data/analysis_ready/`: cleaned and enriched CSV files used by the project notebooks.
- `notebooks/`: project notebooks for analysis and modelling.
- `reference/`: uploaded course/example notebooks used as modelling references.
- `PROJECT_PLAN.md`: review of the uploaded notebooks and how they map to this malaria project.
- `SOURCES.md`: data source notes for the PDFs, GeoJSON, and population denominators.
- `requirements.txt`: Python packages needed for the assignment notebook.

## Generated Data

- `data/processed/district_malaria_2000_2024_from_pdf.csv`
  - District-level malaria cases and deaths from 2000-2024.
  - Includes corrections for merged PDF state cells, including Himachal Pradesh districts that were initially carried under Haryana.

- `data/processed/state_malaria_situation_2021_2025_from_pdf.csv`
  - State/UT-level tested, positive, Pf, and deaths for 2021-2025.

- `data/processed/state_epidemiological_2024_2025_from_pdf.csv`
  - State/UT-level epidemiological indicators for 2024-2025, including Pf%, TPR, TFR, category, imported cases, and indigenous cases where available.

## Usable Datasets

The project notebooks use these analysis-ready CSV files:

- `data/analysis_ready/district_malaria_clean.csv`
- `data/analysis_ready/state_year_malaria_clean.csv`
- `data/analysis_ready/selected_regions_state_year.csv`
- `data/analysis_ready/aiml_state_year_features.csv`
- `data/analysis_ready/data_preparation_report.txt`

## Main Assignment Notebook

Use:

`notebooks/Malaria_Geospatial_Modeling_Assignment.ipynb`

This notebook follows the requested assignment structure:

1. Select 3 regions and spatial scale.
2. Perform mandatory geospatial choropleth analysis before modelling.
3. Fit SIR and ARIMA models separately for Odisha, Mizoram, and Tripura.
4. Add a supervised AI/ML Random Forest model for next-year malaria high-risk classification.

## Standalone AI/ML Notebook

Use:

`notebooks/Malaria_AIML_Risk_Model.ipynb`

This notebook focuses only on the AI/ML model:

- Feature engineering from historical malaria data
- Time-aware train/test split
- Dummy baseline, Logistic Regression, Random Forest, and Gradient Boosting
- Classification metrics, confusion matrix, ROC curve
- Feature importance
- Predictions for Odisha, Mizoram, and Tripura
