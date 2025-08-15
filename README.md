# Shuttle Imbalance Benchmark (GB_base, GB_SMOTE, RF_cw)

Reproduksi eksperimen 5-fold pada **Statlog (Shuttle)** untuk tiga pendekatan:
cost-sensitive Gradient Boosting (GB_base), GB + SMOTE(k=1), dan Random Forest
dengan class_weight. Artefak yang dihasilkan = tabel mean±SD per metrik + figur.

**Dataset:** Statlog (Shuttle) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WS31

## Environment
- Python >= 3.9
- `pip install -r requirements.txt`

## Run
```bash
python scripts/00_pipeline_shuttle_cv.py
