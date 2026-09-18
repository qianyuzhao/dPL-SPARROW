# dPL-SPARROW with G-TREND-N

This repository contains the reproducible code and figure-source data for the
G-TREND-N version of **dPL-SPARROW** (Differentiable Parameter Learning for
SPARROW). The model combines neural parameter generators with physically based
SPARROW routing to estimate annual total nitrogen loads in the Upper Mississippi
River Basin at HUC-12 resolution for 2001–2020.

## Repository contents

```text
01_data_preparation.ipynb       merge SPARROW and G-TREND-N inputs
02_train_dPL_SPARROW.ipynb      train the full 20-year model
03_spatial_validation.ipynb     five-fold spatial validation
04_temporal_validation.ipynb    five-fold temporal validation
05_scenario_attribution.ipynb   early/late-period counterfactual scenarios
06_shap_analysis.ipynb          SHAP analysis of learned parameters
sparrow/                        model and routing utilities
data/publication/               CSV data underlying the manuscript figures
results/validation/             completed validation summaries
environment.yml
```

## Input data

Notebook 01 expects two input tables and five spatial-fold files. Paths are set
in its first code cell.

### SPARROW reach-year table

`data/sparrow_input.csv` has one row per HUC-12 reach and year. Required groups
include network fields (`waterid`, `fnode`, `tnode`, `rchtype`, `headflag`,
`frac`, `iftran`), observed load (`depvar`), routing fields (`strmloss`,
`iresload`), area and hydrology (`demiarea`, `slope`, `meanq`), and the nine
parameter-generator features (`PPT30MEAN`, `tiles_perc`, `soil_CLAYAVE`,
`meanTemp`, `CRP_percent`, `no_till`, `cover_crop_percent`, `forest_percent`,
`wetlands_percent`).

### G-TREND-N table

`data/gtrendn_surplus_1930_2017.csv` has one row per HUC-12 and year. The model
uses `Agriculture_Fertilizer`, `Domestic_Fertilizer`,
`Atmospheric_Oxidized`, `Atmospheric_Reduced`, `Agriculture_Fixation`, `Human`,
`Lvst_Sum`, and `Agriculture_Uptake`. It calculates:

```text
N surplus = fertilizer + deposition + fixation + human + livestock
            - agricultural uptake
total N surplus (kg/yr) = N surplus (kg/ha/yr) × catchment area (ha)
```

G-TREND-N ends in 2017. The preprocessing used for the reported model holds each
catchment's 2017 values constant for 2018, 2019, and 2020.

The large raw model inputs are distributed separately; the compact figure-source
tables are included in `data/publication/`.

## Validation preprocessing

The two validation schemes intentionally use different scaler fitting domains:

- **Temporal validation:** every fold contains 16 training years and four held-out
  years. The main, stream, and reservoir MinMax scalers are fitted only on the 16
  training years, then applied unchanged to both training and validation years.
- **Spatial validation:** the scalers are fitted on covariates for all HUC-12
  reaches, including ungauged and held-out reaches. Routing requires normalized
  covariates throughout the connected network. Validation-site load observations
  remain excluded from model fitting.

The temporal folds are:

| Fold | Held-out years |
|---:|---|
| 1 | 2001, 2006, 2013, 2014 |
| 2 | 2003, 2008, 2012, 2018 |
| 3 | 2004, 2010, 2011, 2016 |
| 4 | 2002, 2015, 2019, 2020 |
| 5 | 2005, 2007, 2009, 2017 |

Both validation notebooks use Adam with learning rate `1e-3`, weight decay
`2e-3`, seed 42, 200 epochs, a 10-epoch linear warmup, and
`StepLR(step_size=50, gamma=0.5)`. The completed temporal training-year-scaler
run reached a five-fold mean validation log-MSE of 0.379796 at epoch 110.

The manuscript figure artifacts were produced from the full-data model's epoch
110 checkpoint (Adam learning rate `1e-3`, weight decay `1e-3`, seed 42).

## Run order

Create the environment and run the notebooks in order:

```bash
conda env create -f environment.yml
conda activate dpl-sparrow
jupyter lab
```

```text
01 → 02 → 03 and 04 → 05 → 06
```

GPU execution is recommended for notebooks 02–04. Data preparation and analysis
can run on CPU.

## Publication data

The files in `data/publication/` contain the data underlying spatial maps,
scenario comparisons, observed-versus-predicted plots, residual maps, and SHAP
plots. `data/publication/DATA_DICTIONARY.md` documents every table.

The data-generation script is retained with the project files on Delta rather
than distributed in this GitHub repository.

## Contact

Questions: qz29@illinois.com, binpeng@illinois.edu, kaiyug@illinois.edu
