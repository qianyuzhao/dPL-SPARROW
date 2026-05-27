# dPL-SPARROW: Differentiable Parameter Learning for SPARROW Nitrogen Load Modeling

This repository contains the code for training and evaluating the **dPL-SPARROW** model, which combines the physically-based SPARROW (SPAtially Referenced Regressions On Watershed attributes) stream load model with neural networks (*ParamGenerators*) that learn spatially varying parameters from landscape attributes.

The model is applied to the **Upper Mississippi River Basin (UMRB)** at the HUC-12 scale to simulate annual total nitrogen (TN) loads (2001–2020), attribute loads to N-budget components, and explain parameter variability with SHAP values.

---

## Repository structure

```
├── sparrow/
│   ├── model.py          SPARROW routing module and ParamGenerator MLP
│   └── utils.py          hydseq, gpu_align, r2_torch, r2_logspace_torch, setup_seed
│
├── 01_data_preparation.ipynb      Load SPARROW input + TREND-N data, remap nodes, compute HYDSEQ
├── 02_train_dPL_SPARROW.ipynb     Train dPL-SPARROW on the full 20-year dataset
├── 03_spatial_validation.ipynb    5-fold spatial cross-validation
├── 04_temporal_validation.ipynb   5-fold temporal cross-validation
├── 05_scenario_attribution.ipynb  N-budget component attribution (proportional apportionment)
├── 06_shap_analysis.ipynb         DeepSHAP analysis of learned SPARROW parameters
│
├── environment.yml                Conda environment specification
└── README.md
```

---

## Required data

### 1. SPARROW input table

One row per **HUC-12 reach × year**.  Set `sparrow_input_path` in notebook 01 to point to this file.  Required columns:

| Column group | Columns | Description |
|---|---|---|
| Network topology | `waterid`, `fnode`, `tnode`, `rchtype`, `headflag`, `frac`, `iftran` | Stream network connectivity |
| Observation | `depvar` | Annual TN load at gauging stations (kg/yr); 0 = ungauged |
| Stream / reservoir | `strmloss`, `iresload` | In-stream decay and reservoir retention variables |
| Hydrology / terrain | `slope`, `meanq`, `demiarea` | Per-reach physical attributes |
| Climate | `PPT`, `meanTemp` | Annual total precipitation and mean air temperature |
| Soil / land use | `tiles_perc`, `soil_CLAYAVE`, `CRP_percent`, `no_till`, `cover_crop_percent`, `forest_percent`, `wetlands_percent` | Landscape controls on N export |
| Year | `Year` | Calendar year (integer) |

### 2. TREND-N nitrogen surplus data

Annual N-budget variables from the TREND-N model, one row per HUC-12 reach × year.  Set `trendn_path` in notebook 01 to point to this file.

Required columns: `waterid`, `Year`, `NSurplus` (kg N ha⁻¹ yr⁻¹), plus the N input and uptake components used for scenario attribution: `Atmospheric_Oxidized`, `Atmospheric_Reduced`, `Fertilizer_Agriculture`, `Fertilizer_NonAgriculture`, `Fix_Cropland`, `Fix_Pasture`, `Human`, livestock columns (`Lvst_BeefCow` … `Lvst_Turkeys`), `CropUptake_Cropland`, `CropUptake_Pasture`.

The notebooks compute `total_N_surplus = N_surplus × area_ha` (kg yr⁻¹) as the single N source for SPARROW routing.

### 3. Spatial fold assignments (for notebook 03)

Five CSV files, one per fold (`CV_1_data.csv` … `CV_5_data.csv`).  Set `cv_fold_dir` in notebook 03 to point to the folder containing them.

Each file must have a `waterid` column and a `valsites` column (1 = validation reach for that fold).

### 4. Temporal fold assignments (for notebook 04)

Pre-defined in code — no additional files needed.

The four held-out years per fold were drawn by random selection from the 20-year study period, stratified to spread held-out years across the full time range rather than grouping them consecutively.

| Fold | Held-out years |
|---|---|
| 1 | 2001, 2006, 2013, 2014 |
| 2 | 2003, 2008, 2012, 2018 |
| 3 | 2004, 2010, 2011, 2016 |
| 4 | 2002, 2015, 2019, 2020 |
| 5 | 2005, 2007, 2009, 2017 |

### Data availability

The input data are hosted on [Figshare – add DOI here].  Download and unzip to a local folder, then set `working_dir` accordingly.

---

## Installation

```bash
conda env create -f environment.yml
conda activate dpl-sparrow
```

For GPU training (recommended), ensure your system has CUDA installed and that the PyTorch version in `environment.yml` matches your CUDA version.  See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for CUDA-specific install commands.

---

## How to run

Run the notebooks **in order**:

```
01 → 02 → 03 / 04 → 05 → 06
```

Each notebook writes its outputs to a sub-folder under `./outputs/`.  The first cell of every notebook contains a `data_dir` variable — set this to the root of your data folder before running.

| Notebook | Approximate runtime | GPU required? |
|---|---|---|
| 01 Data preparation | ~5 min | No |
| 02 Full training (150 epochs) | ~2 h | Recommended |
| 03 Spatial CV (5 × 150 epochs) | ~10 h | Recommended |
| 04 Temporal CV (5 × 150 epochs) | ~10 h | Recommended |
| 05 Scenario attribution | ~15 min | No |
| 06 SHAP analysis | ~15 min | No |

Training can be accelerated by reducing `NUM_EPOCHS` for exploratory runs.

---

## Model overview

```
Catchment attributes (9 features)         Stream attrs (slope, meanq)   Reservoir (meanTemp)
         │                                         │                           │
  ┌──────▼──────┐                          ┌───────▼──────┐           ┌───────▼──────┐
  │ param_model │  hidden=32               │param_model_  │ hidden=8  │param_model_  │ hidden=8
  │  (MLP, 9→32)│                          │strm (MLP,2→8)│           │res (MLP,1→8) │
  └──────┬──────┘                          └──────┬───────┘           └──────┬───────┘
         │  α (export), θ_D (delivery)            │  θ_S (stream loss)       │  θ_R (res. loss)
         └────────────────────┬───────────────────┘───────────────────────────┘
                              │  spatially varying parameters per reach
                              ▼
                    ┌──────────────────┐
                    │  SPARROW routing │  physics-based, upstream → downstream
                    └────────┬─────────┘
                             │
                  Predicted TN load at each reach
```

Three `ParamGenerator` MLPs are trained jointly with a single Adam optimizer:

| Sub-network | Inputs | Hidden | Outputs used |
|---|---|---|---|
| `param_model` | 9 catchment attributes | 32 | α (N export), θ_D (delivery) |
| `param_model_strm` | slope, meanq | 8 | θ_S = `coeffs[:, -2:-1]` |
| `param_model_res` | meanTemp | 8 | θ_R = `coeffs[:, -1:]` |

**Single N source:** `total_N_surplus` (TREND-N N surplus × catchment area).

**Training loss:** MSE in log-space between predicted and observed annual TN loads at gauging stations.

---

## Contact

Questions contact: qz29@illinois.com, binpeng@illinois.edu, kaiyug@illinois.edu
