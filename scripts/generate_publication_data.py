"""
generate_figshare_csvs.py
--------------------------
Generates the organized G-TREND-N CSV data files for publication.
These CSVs contain the underlying data used to plot all figures in:
  /work/hdd/bbkc/qz29/nwis/differentiable_sparrow/figures_cover_crop/

Source notebooks are in
  /u/qz29/SPARROW_DL/gpu_train/simulation_results_summary/gtrendn_results/
  human_hydro_spatial_grey_background.ipynb  -> 05_scenario_N_export_rate_by_HUC12.csv
  input_spatial_grey_bg.ipynb               -> 04_input_landuse_climate_*.csv
  load_yields.ipynb                          -> 01_N_export_load_yield_by_HUC12_period.csv
  Residual_map_sim.ipynb                     -> 07_residuals_at_monitoring_watersheds.csv
  SHAP-Copy2.ipynb                           -> 08_SHAP_input_features_by_HUC12_year.csv
  sim_UMRB.ipynb                             -> 06_observed_vs_predicted_load_monitoring_sites.csv
  spatial_20yr.ipynb                         -> 01, 02 (also used)
  spatial.ipynb                              -> 02_model_parameters_by_HUC12_year.csv
  surplus_spatial_grey_background.ipynb      -> 03_N_surplus_by_HUC12_period.csv

Run with:
  /u/qz29/pyenvqz/pygeo_env/bin/python generate_figshare_csvs.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
base = os.environ.get('DPL_SPARROW_DATA_ROOT', '/work/hdd/bbkc/qz29/nwis/').rstrip('/') + '/'
save_dir_coeff = (
    base + 'differentiable_sparrow/train_dPL/'
    'input_select_train_huc12_trend_test31_50step_cover_crop_bnf_source_qavg_gtrend/'
)
out_dir = os.environ.get(
    'DPL_SPARROW_PUBLICATION_DATA',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'publication'),
).rstrip('/') + '/'
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------
# Load base datasets (shared across multiple CSVs)
# ---------------------------------------------------------------
print("Loading base datasets...")

sim_yield = pd.read_csv(save_dir_coeff + 'simulation_results_yield.csv')
sim_yield['Year'] = ('20' + sim_yield['waterid'].astype(int).astype(str).str[-2:]).astype(int)
sim_yield['HUC_12'] = sim_yield['waterid'].astype(int).astype(str).str[:-2].astype(int)

coeffs = pd.read_csv(save_dir_coeff + 'generated_coefficients_with_HUC10_Year_static.csv')

input_data = pd.read_csv(
    base + 'datasets/UMRB_data_SPARROW/HUC12_scale/SPARROW_input/'
    'HUC12_input_2001_2020_cdl_urban_mean_adjusted_all_hf_updated_gSSURGO.csv'
)
input_data = input_data.fillna(0)

surplus = pd.read_csv(
    base + 'datasets/UMRB_data_SPARROW/HUC12_scale/N_surplus/'
    'gtrend_N_surplus_1930_2017.csv'
)
surplus['HUC_12'] = surplus['HUC_12'].astype(str)

# The G-TREND-N source ends in 2017. Match the model preprocessing by holding
# the 2017 source values constant for 2018--2020 before constructing waterid.
surplus_2017 = surplus[surplus['year'] == 2017].copy()
future = []
for year in (2018, 2019, 2020):
    year_data = surplus_2017.copy()
    year_data['year'] = year
    future.append(year_data)
surplus = pd.concat([surplus] + future, ignore_index=True)
surplus['waterid'] = (
    surplus['HUC_12'] + surplus['year'].astype(str).str[-2:]
).astype(int)

print("  Done.")


def period_label(year):
    if year <= 2005:
        return '2001-2005'
    elif year >= 2016:
        return '2016-2020'
    else:
        return 'other'


# ---------------------------------------------------------------
# CSV 01: Simulation results (load & yield) by HUC-12 and period
# ---------------------------------------------------------------
# Source notebooks: load_yields.ipynb, spatial_20yr.ipynb
# Source file: simulation_results_yield.csv
# Figures: FigureS6_N_load_yield_Map (total_load -> log10 color, inc_yield_wo_decay / 100)
# ---------------------------------------------------------------
print("Creating 01_N_export_load_yield_by_HUC12_period.csv ...")

sim_yield['period'] = sim_yield['Year'].apply(period_label)
sim_period = sim_yield[sim_yield['period'] != 'other'].copy()
sim_period['inc_yield_wo_decay_kg_ha_yr'] = sim_period['inc_yield_wo_decay'] / 100

agg1 = sim_period.groupby(['HUC_12', 'period']).agg(
    total_load_kg_yr=('total_load', 'mean'),
    inc_load_kg_yr=('inc_load', 'mean'),
    inc_yield_wo_decay_kg_ha_yr=('inc_yield_wo_decay_kg_ha_yr', 'mean'),
    delivery_fraction=('delivery_fraction', 'mean'),
    stream_attenuation_local=('local_att', 'mean'),
    watershed_attenuation_total=('total_att', 'mean'),
    watershed_area_km2=('demiarea', 'mean'),
).reset_index()

agg1.to_csv(out_dir + '01_N_export_load_yield_by_HUC12_period.csv', index=False)
print(f"  Saved: {agg1.shape}")


# ---------------------------------------------------------------
# CSV 02: Model parameters by HUC-12 and year (2001-2020)
# ---------------------------------------------------------------
# Source notebooks: spatial.ipynb, spatial_20yr.ipynb,
#                   human_hydro_spatial_grey_background.ipynb
# Source files: generated_coefficients_with_HUC10_Year_static.csv
#               + HUC12_input_...csv (for iresload, strmloss)
# Figures: FigureS4_Stream_parameter_Map, FigureS5_Reservoir_parameter_Map,
#          FigureS5_N_loss_to_water_ratio_Map, Figure2/4/6 spatial maps,
#          FigureS6_Stream_SHAP, FigureS7_Reservoir_SHAP
# ---------------------------------------------------------------
print("Creating 02_model_parameters_by_HUC12_year.csv ...")

# Merge to compute derived attenuation columns:
#   Reservoir_Attenuation = Reservoir Loss * iresload
#   Stream_Attenuation    = Stream Loss    * strmloss
merged_params = coeffs.merge(
    input_data[['waterid', 'iresload', 'strmloss']],
    left_on='original_catchment_id', right_on='waterid', how='left'
)
merged_params['Reservoir_Attenuation'] = merged_params['Reservoir Loss'] * merged_params['iresload']
merged_params['Stream_Attenuation'] = merged_params['Stream Loss'] * merged_params['strmloss']

out_params = merged_params[[
    'HUC_12', 'Year',
    'N Export Rate', 'Stream Loss', 'Reservoir Loss',
    'Reservoir_Attenuation', 'Stream_Attenuation',
]].copy()
out_params.columns = [
    'HUC_12', 'Year',
    'N_Export_Rate', 'Stream_Loss', 'Reservoir_Loss',
    'Reservoir_Attenuation', 'Stream_Attenuation',
]

out_params.to_csv(out_dir + '02_model_parameters_by_HUC12_year.csv', index=False)
print(f"  Saved: {out_params.shape}")


# ---------------------------------------------------------------
# CSV 03: G-TREND-N surplus by HUC-12 and period
# ---------------------------------------------------------------
# Source notebook: surplus_spatial_grey_background.ipynb
# Source files: HUC12_TRENDN_00_20.csv merged with input data for Year column
# Figures: Figure_S11_*_surpus_change_map (differences between periods)
# ---------------------------------------------------------------
print("Creating 03_N_surplus_by_HUC12_period.csv ...")

input_surplus = input_data[['waterid', 'Year']].merge(
    surplus.drop(columns=['HUC_12']), on='waterid', how='left'
)

# These definitions exactly match the G-TREND-N model preprocessing.
input_surplus['Manure_surplus'] = input_surplus['Lvst_Sum']
input_surplus['Fert_surplus'] = (
    input_surplus['Agriculture_Fertilizer'] + input_surplus['Domestic_Fertilizer']
)
input_surplus['Fix_surplus'] = input_surplus['Agriculture_Fixation']
input_surplus['ndep_surplus'] = (
    input_surplus['Atmospheric_Oxidized'] + input_surplus['Atmospheric_Reduced']
)
input_surplus['Total_Uptake'] = input_surplus['Agriculture_Uptake']
input_surplus['N_surplus'] = (
    input_surplus['Agriculture_Fertilizer']
    + input_surplus['Domestic_Fertilizer']
    + input_surplus['Atmospheric_Oxidized']
    + input_surplus['Atmospheric_Reduced']
    + input_surplus['Agriculture_Fixation']
    + input_surplus['Human']
    + input_surplus['Lvst_Sum']
    - input_surplus['Agriculture_Uptake']
)

input_surplus['HUC_12'] = input_surplus['waterid'].astype(str).str[:-2].astype(int)
input_surplus['period'] = input_surplus['Year'].apply(period_label)
input_surplus_period = input_surplus[input_surplus['period'] != 'other'].copy()

agg3 = input_surplus_period.groupby(['HUC_12', 'period']).agg(
    N_surplus_kgN_ha_yr=('N_surplus', 'mean'),
    Manure_surplus_kgN_ha_yr=('Manure_surplus', 'mean'),
    Fertilizer_surplus_kgN_ha_yr=('Fert_surplus', 'mean'),
    Fix_surplus_kgN_ha_yr=('Fix_surplus', 'mean'),
    ndep_surplus_kgN_ha_yr=('ndep_surplus', 'mean'),
    Human_kgN_ha_yr=('Human', 'mean'),
    Total_Uptake_kgN_ha_yr=('Total_Uptake', 'mean'),
).reset_index()

agg3.to_csv(out_dir + '03_N_surplus_by_HUC12_period.csv', index=False)
print(f"  Saved: {agg3.shape}")


# ---------------------------------------------------------------
# CSV 04 & 04b: Input land use and climate by HUC-12 and period
# ---------------------------------------------------------------
# Source notebook: input_spatial_grey_bg.ipynb
# Source file: generated_coefficients_with_HUC10_Year_static.csv
# Figures: FigureS10 change maps, FigureS12 forest/wetlands maps,
#          FigureS18_cover_crop_Map, FigureS23_meanTemp_Map,
#          Figure_S10_tiles_perc, Figure_S10_soil_clay
# ---------------------------------------------------------------
print("Creating 04_input_landuse_climate_by_HUC12_period.csv ...")

coeffs['period'] = coeffs['Year'].apply(period_label)
coeffs_period = coeffs[coeffs['period'] != 'other'].copy()

agg4 = coeffs_period.groupby(['HUC_12', 'period']).agg(
    cover_crop_percent=('cover_crop_percent', 'mean'),
    tiles_perc=('tiles_perc', 'mean'),
    wetlands_percent=('wetlands_percent', 'mean'),
    forest_percent=('forest_percent', 'mean'),
    CRP_percent=('CRP_percent', 'mean'),
    no_till_fraction=('no_till', 'mean'),
    PPT30MEAN_mm=('PPT30MEAN', 'mean'),
    meanTemp_degC=('meanTemp', 'mean'),
    soil_CLAYAVE_fraction=('soil_CLAYAVE', 'mean'),
).reset_index()

agg4.to_csv(out_dir + '04_input_landuse_climate_by_HUC12_period.csv', index=False)
print(f"  Saved 04: {agg4.shape}")

# 04b: wide format with pre-computed period differences
print("Creating 04b_input_landuse_climate_change_HUC12.csv ...")

pivot4 = agg4.pivot(index='HUC_12', columns='period', values=[
    'cover_crop_percent', 'tiles_perc', 'wetlands_percent', 'forest_percent',
    'CRP_percent', 'no_till_fraction', 'PPT30MEAN_mm', 'meanTemp_degC', 'soil_CLAYAVE_fraction'
])
pivot4.columns = ['_'.join(c) for c in pivot4.columns]
pivot4 = pivot4.reset_index()

for var in ['cover_crop_percent', 'wetlands_percent', 'forest_percent',
            'CRP_percent', 'no_till_fraction', 'PPT30MEAN_mm', 'meanTemp_degC']:
    c2001 = f'{var}_2001-2005'
    c2016 = f'{var}_2016-2020'
    if c2001 in pivot4.columns and c2016 in pivot4.columns:
        pivot4[f'{var}_diff_2016minus2001'] = pivot4[c2016] - pivot4[c2001]

pivot4.to_csv(out_dir + '04b_input_landuse_climate_change_HUC12.csv', index=False)
print(f"  Saved 04b: {pivot4.shape}")


# ---------------------------------------------------------------
# CSV 05: Scenario N Export Rate comparison by HUC-12
# ---------------------------------------------------------------
# Source notebook: human_hydro_spatial_grey_background.ipynb, spatial.ipynb
# Source files: generated_coefficients_2001_2005_[scenario].csv
# Figures: Figure_S8–S16, Figure_S12_hydro_human_* panels,
#          hydro_human_2panel_* figures
#
# Each scenario file holds counterfactual model outputs where one set of
# input features is frozen at 2001-2005 levels while others advance to 2016-2020.
# The sequential difference between files isolates each factor's contribution.
# ---------------------------------------------------------------
print("Creating 05_scenario_N_export_rate_by_HUC12.csv ...")

scenario_files = {
    'baseline':      'generated_coefficients_with_HUC10_Year_static.csv',
    'cover_crop_2001':   'generated_coefficients_2001_2005_cover_crop.csv',
    'wetlands_2001':     'generated_coefficients_2001_2005_cover_crop_wetlands.csv',
    'forest_2001':       'generated_coefficients_2001_2005_cover_crop_wetlands_forest.csv',
    'crp_2001':          'generated_coefficients_2001_2005_cover_crop_wetlands_forest_crp.csv',
    'human_prec_2001':   'generated_coefficients_2001_2005_human_prec.csv',
    'hydro_2001':        'generated_coefficients_2001_2005_hydro.csv',
}

dfs = {}
for scenario, fname in scenario_files.items():
    df = pd.read_csv(save_dir_coeff + fname)
    if scenario == 'baseline':
        df_01 = df[df['Year'] <= 2005]
        df_16 = df[df['Year'] >= 2016]
        dfs['baseline_2001_2005'] = (
            df_01.groupby('HUC_12')['N Export Rate'].mean().reset_index()
            .rename(columns={'N Export Rate': 'N_Export_Rate_baseline_2001_2005'})
        )
        dfs['baseline_2016_2020'] = (
            df_16.groupby('HUC_12')['N Export Rate'].mean().reset_index()
            .rename(columns={'N Export Rate': 'N_Export_Rate_baseline_2016_2020'})
        )
    else:
        dfs[scenario] = (
            df.groupby('HUC_12')['N Export Rate'].mean().reset_index()
            .rename(columns={'N Export Rate': f'N_Export_Rate_{scenario}'})
        )

df5 = dfs['baseline_2001_2005']
for k, v in dfs.items():
    if k != 'baseline_2001_2005':
        df5 = df5.merge(v, on='HUC_12', how='outer')

df5.to_csv(out_dir + '05_scenario_N_export_rate_by_HUC12.csv', index=False)
print(f"  Saved: {df5.shape}")


# ---------------------------------------------------------------
# CSV 06: Observed vs. predicted load at monitoring sites
# ---------------------------------------------------------------
# Source notebook: sim_UMRB.ipynb
# Source files: simulation_results.csv + HUC12_input_...csv
# Figures: FigureS5_scatter_map (hexbin, log10 scale)
# Only rows where depvar > 0 (WRTDS-calibrated monitoring sites)
# ---------------------------------------------------------------
print("Creating 06_observed_vs_predicted_load_monitoring_sites.csv ...")

sim_results = pd.read_csv(save_dir_coeff + 'simulation_results.csv')
scatter = sim_results.merge(
    input_data[['waterid', 'depvar', 'demiarea', 'demtarea']],
    on='waterid', how='inner'
)
scatter = scatter[scatter['depvar'] > 0].copy()
scatter['HUC_12'] = scatter['waterid'].astype(str).str[:-2].astype(int)
scatter['Year'] = scatter['waterid'].astype(str).str[-2:].astype(int) + 2000
scatter['observed_load_kg_yr'] = scatter['depvar']
scatter['predicted_load_kg_yr'] = scatter['total_load']
scatter['observed_yield_kg_km2_yr'] = scatter['depvar'] / scatter['demiarea']
scatter['predicted_yield_kg_km2_yr'] = scatter['total_load'] / scatter['demiarea']

out6 = scatter[[
    'HUC_12', 'waterid', 'Year',
    'observed_load_kg_yr', 'predicted_load_kg_yr',
    'observed_yield_kg_km2_yr', 'predicted_yield_kg_km2_yr',
    'demiarea', 'demtarea'
]]
out6.to_csv(out_dir + '06_observed_vs_predicted_load_monitoring_sites.csv', index=False)
print(f"  Saved: {out6.shape}")


# ---------------------------------------------------------------
# CSV 07: Residuals at monitoring watersheds
# ---------------------------------------------------------------
# Source notebook: Residual_map_sim.ipynb
# Source files: SPARROW_sites_inc_gdf_wrtdsk (shapefile) - dPL model predictions
#               WRTDS_sites_inc_gdf_1 (shapefile) - WRTDS observed loads
# Figures: FigureS4_Residual_Yield_Map
#
# Residual formula (from notebook):
#   resid = (WRTDS_obs - model_pred) / (inc_area_miles2 * 2.58999 * 100)
#   Units: kg/ha/yr
# ---------------------------------------------------------------
print("Creating 07_residuals_at_monitoring_watersheds.csv ...")

inc_sparrow = gpd.read_file(
    base + 'datasets/UMRB_data_SPARROW/HUC12_shape/SPARROW_sites_inc_gdf_wrtdsk'
)
inc_sparrow = inc_sparrow.rename(columns={
    'inc_load_2': 'inc_load_2001_2005',
    'inc_load_1': 'inc_load_2016_2020',
    'inc_area_k': 'inc_area_km2',
    'inc_yield_': 'inc_yield_2001_2005',
    'inc_yiel_1': 'inc_yield_2016_2020',
    'inc_yiel_2': 'inc_yield_diff',
})

inc_wrtds = gpd.read_file(
    base + 'datasets/UMRB_data_SPARROW/HUC12_shape/WRTDS_sites_inc_gdf_1'
)
inc_wrtds = inc_wrtds.rename(columns={
    'inc_load_2': 'inc_load_2001_2005_Gen',
    'inc_load_1': 'inc_load_2016_2020_Gen',
    'inc_area_k': 'inc_area_km2',
    'Q_2001_200': 'Q_2001_2005',
    'Q_2016_202': 'Q_2016_2020',
})

merged_res = inc_sparrow[[
    'site_no', 'Label', 'inc_area', 'inc_area_km2',
    'inc_load_2001_2005', 'inc_load_2016_2020',
    'inc_yield_2001_2005', 'inc_yield_2016_2020'
]].merge(
    inc_wrtds[[
        'site_no', 'inc_load_2001_2005_Gen', 'inc_load_2016_2020_Gen',
        'Q_2001_2005', 'Q_2016_2020'
    ]],
    on='site_no', how='inner'
)

# Residual formula from Residual_map_sim.ipynb cell 16/18:
#   resid = (obs - pred) / (inc_area [miles²] * 2.58999 [km²/mi²] * 100 [ha/km²])
merged_res['residual_yield_2001_2005_kg_ha_yr'] = (
    (merged_res['inc_load_2001_2005_Gen'] - merged_res['inc_load_2001_2005'])
    / (merged_res['inc_area'] * 2.58999 * 100)
)
merged_res['residual_yield_2016_2020_kg_ha_yr'] = (
    (merged_res['inc_load_2016_2020_Gen'] - merged_res['inc_load_2016_2020'])
    / (merged_res['inc_area'] * 2.58999 * 100)
)

out7 = merged_res.copy()
out7.columns = [
    'site_no', 'watershed_label',
    'watershed_area_miles2', 'watershed_area_km2',
    'SPARROW_static_inc_load_2001_2005_kg_yr', 'dPL_inc_load_2016_2020_kg_yr',
    'SPARROW_static_inc_yield_2001_2005_kg_km2_yr', 'dPL_inc_yield_2016_2020_kg_km2_yr',
    'WRTDS_inc_load_2001_2005_kg_yr', 'WRTDS_inc_load_2016_2020_kg_yr',
    'mean_discharge_2001_2005_m3_s', 'mean_discharge_2016_2020_m3_s',
    'residual_yield_2001_2005_kg_ha_yr', 'residual_yield_2016_2020_kg_ha_yr',
]

out7.to_csv(out_dir + '07_residuals_at_monitoring_watersheds.csv', index=False)
print(f"  Saved: {out7.shape}")


# ---------------------------------------------------------------
# CSV 08: SHAP values for N Export Rate (from shap_additive_model.ipynb)
# ---------------------------------------------------------------
# Source notebook: shap_additive_model.ipynb
# Source files: shap_values_all.npy, X_sample_subset.npy, sample_idx.npy
#               (saved by the notebook to save_dir_coeff, without trailing /)
# Figures: Figure5_Loss_Ratio_SHAP (beeswarm / bar plot)
#
# DeepSHAP was run on 20,000 random samples from the full dataset (116,360 rows).
# sample_idx maps each of the 20,000 rows back to the full input dataset ordered
# by Year (in order of appearance: 2001..2020), then by HUC_12 within each year.
#
# Feature order in shap_values_all[:, :, 0] (N Export Rate output):
#   0: Precipitation (PPT30MEAN)      5: No-Till (no_till)
#   1: Tile Drainage (tiles_perc)     6: Cover Crop (cover_crop_percent)
#   2: Clay Soil (soil_CLAYAVE)       7: Forest (forest_percent)
#   3: Air Temperature (meanTemp)     8: Wetlands (wetlands_percent)
#   4: CRP (CRP_percent)
# ---------------------------------------------------------------
print("Creating 08_SHAP_values_N_export_rate.csv ...")

from sklearn.preprocessing import MinMaxScaler

npy_base = save_dir_coeff

shap_values_all = np.load(os.path.join(npy_base, 'shap_values_all.npy'))
X_sample_subset = np.load(os.path.join(npy_base, 'X_sample_subset.npy'))
sample_idx      = np.load(os.path.join(npy_base, 'sample_idx.npy'))

# Reconstruct the same row ordering used when building X_all_scaled_np:
# iterate input_data['Year'].unique() (order of first appearance) and
# stack year-wise blocks in that order.
input_columns_shap = ['PPT30MEAN', 'tiles_perc', 'soil_CLAYAVE', 'meanTemp',
                      'CRP_percent', 'no_till', 'cover_crop_percent',
                      'forest_percent', 'wetlands_percent']

scaler_shap = MinMaxScaler().fit(input_data[input_columns_shap])
years_order = input_data['Year'].unique()

wid_chunks = []
for year in years_order:
    yd = input_data[input_data['Year'] == year]
    wid_chunks.append(yd[['waterid', 'HUC_12', 'Year']].values)
wid_all = np.vstack(wid_chunks)   # (116360, 3), same order as X_all_scaled_np

# Select the 20,000 sampled rows
wid_sampled = wid_all[sample_idx]   # (20000, 3)

# SHAP values for N Export Rate (output index 0)
shap_vals_nexport = shap_values_all[:, :, 0]   # (20000, 9)

feature_display_names = [
    'Precipitation_mm', 'Tile_Drainage_pct', 'Clay_Soil_pct',
    'Air_Temperature_degC', 'CRP_pct', 'No_Till_pct',
    'Cover_Crop_pct', 'Forest_pct', 'Wetlands_pct'
]

df8 = pd.DataFrame({
    'waterid': wid_sampled[:, 0],
    'HUC_12':  wid_sampled[:, 1],
    'Year':    wid_sampled[:, 2],
})

# Raw (unscaled) feature values
raw_features = scaler_shap.inverse_transform(X_sample_subset)
for j, fname in enumerate(feature_display_names):
    df8[f'feature_{fname}'] = raw_features[:, j]

# SHAP values
for j, fname in enumerate(feature_display_names):
    df8[f'SHAP_{fname}'] = shap_vals_nexport[:, j]

df8.to_csv(out_dir + '08_SHAP_values_N_export_rate.csv', index=False)
print(f"  Saved: {df8.shape}")


print("\nAll done. Files written to:")
print(f"  {out_dir}")
for f in sorted(os.listdir(out_dir)):
    size_mb = os.path.getsize(out_dir + f) / 1e6
    print(f"  {f:55s} {size_mb:6.1f} MB")
