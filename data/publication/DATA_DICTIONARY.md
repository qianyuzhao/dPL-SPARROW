# Publication data dictionary

All identifiers are HUC-12 codes unless a monitoring-site identifier is shown.
Period values are five-year means for 2001–2005 or 2016–2020. G-TREND-N values
for 2018–2020 use the model's documented 2017-value extension.

| File | Rows | Contents |
|---|---:|---|
| `01_N_export_load_yield_by_HUC12_period.csv` | 11,636 | Routed total and incremental loads, incremental yield, delivery fraction, attenuation, and watershed area by HUC-12 and period. |
| `02_model_parameters_by_HUC12_year.csv` | 116,360 | Annual learned land-to-stream delivery efficiency, stream-loss, and reservoir-loss parameters plus derived attenuation. |
| `03_N_surplus_by_HUC12_period.csv` | 11,636 | G-TREND-N surplus and its manure, fertilizer, fixation, deposition, human, and uptake components in kg N ha⁻¹ yr⁻¹. |
| `04_input_landuse_climate_by_HUC12_period.csv` | 11,636 | Period means of land-management, land-cover, climate, and soil inputs. |
| `04b_input_landuse_climate_change_HUC12.csv` | 5,818 | Wide-format early/late input means and late-minus-early changes. |
| `05_scenario_land_to_stream_delivery_efficiency_by_HUC12.csv` | 5,818 | Baseline and sequential counterfactual land-to-stream delivery-efficiency scenarios. Scenario labels identify inputs held at early-period values. |
| `06_observed_vs_predicted_load_monitoring_sites.csv` | 2,080 | Annual observed and modeled loads and yields at monitoring sites. |
| `07_residuals_at_monitoring_watersheds.csv` | 104 | Early/late incremental loads, yields, discharge, and residual yields for monitoring watersheds. |
| `08_SHAP_values_land_to_stream_delivery_efficiency.csv` | 20,000 | Sample identifiers, raw feature values, and SHAP values for learned land-to-stream delivery efficiency. |

Column names contain units where applicable. Percent/fraction fields retain the
units of the model input table. `waterid` combines the HUC-12 identifier with a
two-digit year suffix; `HUC_12` contains the catchment identifier alone.

The CSVs were generated from the archived G-TREND-N model artifacts on Delta.
