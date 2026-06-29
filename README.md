# Pulse Excitation Informativeness for Battery Capacity Diagnosis

This repository contains the code accompanying the study:

**Constant-flux diffusion scaling maps pulse-excitation informativeness for battery capacity diagnosis**

The study maps how the diagnostic informativeness of pulse excitation depends on pulse current intensity, pulse duration, and charge status for battery capacity diagnosis. We use lithium iron phosphate (LFP) cells with nominal capacities of 20 Ah, 35 Ah, and 68 Ah, and evaluate both single-pulse excitation and combinational-pulse excitation protocols. Diagnosis error is used as the measure of pulse-excitation informativeness, and a diffusion-scaled descriptor derived from constant-flux diffusion scaling is used to organize the observed error landscapes.

The code is organized to mirror the data workflow used in the manuscript:

```text
Raw pulse-excitation data
-> step-1 workstep extraction
-> step-2 pulse-feature extraction
-> step-3 model-input feature table construction
-> single-pulse and combinational-pulse model experiments
-> aggregated model-result analysis
-> main-figure source data and plotting
```

## Repository Structure

```text
.
|-- 01_Data_preprocessing_codes
|   |-- PulseBat_raw_to_model_features
|   |   |-- step_1_extract workstep sheet.py
|   |   |-- step_2_feature extraction_adjustable.py
|   |   |-- step_2_feature extraction_complete.py
|   |   |-- step_3_feature collection_adjustable.py
|   |   `-- step_3_feature collection_complete.py
|   `-- Auxiliary_main_figure_data_extraction
|       `-- extract_processdata_by_rows.py
|-- 02_Model_experiment_codes
|   |-- single_feature_experiments
|   |   |-- model.py
|   |   |-- experiment.py
|   |   `-- run.sbatch
|   `-- combinational_feature_experiments
|       |-- model.py
|       |-- experiment.py
|       `-- run.sbatch
|-- 03_Main_figure_plotting_codes
|   `-- PulseBat_combo_main_figures
|       |-- Figure1
|       |   |-- Figure1a/Figure1a.py
|       |   |-- Figure1b/Figure1b.py
|       |   |-- Figure1c/Figure1c.py
|       |   `-- Figure1d/Figure1d.py
|       |-- Figure2
|       |   |-- Figure2a/Figure2a.py
|       |   |-- Figure2a/Figure2a_zoom.py
|       |   |-- Figure2b/Figure2b.py
|       |   |-- Figure2c/Figure2c.py
|       |   |-- Figure2c/Figure2c_colorbar.py
|       |   `-- Figure2d/Figure2d.py
|       |-- Figure3
|       |   |-- Figure3a/Figure3a.py
|       |   |-- Figure3b/Figure3b.py
|       |   |-- Figure3c/Figure3c.py
|       |   |-- Figure3d/Figure3d.py
|       |   |-- Figure3e/Figure3e.py
|       |   `-- Figure3f/Figure3f.py
|       |-- Figure4
|       |   |-- Figure4a/Figure4a.py
|       |   |-- Figure4b/Figure4b.py
|       |   |-- Figure4c/Figure4c.py
|       |   |-- Figure4d/Figure4d.py
|       |   |-- Figure4e/Figure4e.py
|       |   |-- Figure4extra/Figure4extra.py
|       |   `-- Figure4f/Figure4f.py
|       `-- Figure5
|           |-- figure5_common.py
|           |-- Figure5a/Figure5a.py
|           |-- Figure5b/Figure5b.py
|           |-- Figure5c/Figure5c.py
|           |-- Figure5f/Figure5f.py
|           |-- Figure5g/Figure5g.py
|           `-- run_all_figure5.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Code Organization

### 01_Data_preprocessing_codes

Scripts for transforming raw pulse-excitation Excel files into model-ready feature tables.

`PulseBat_raw_to_model_features` contains the main preprocessing pipeline:

```text
step_1_extract workstep sheet.py
step_2_feature extraction_adjustable.py
step_2_feature extraction_complete.py
step_3_feature collection_adjustable.py
step_3_feature collection_complete.py
```

These scripts correspond to the preprocessing stages in the associated dataset:

```text
01_RawData_LFP
02_Preprocessed_step1_workstep_sheet
03_Preprocessed_step2_feature_extraction
04_Model_Input_Feature_Tables
```

`step_1_extract workstep sheet.py` extracts workstep-level records from raw experimental files. `step_2_feature extraction_*.py` extracts pulse-response features from voltage responses. `step_3_feature collection_*.py` collects extracted features into final model-input feature tables.

`Auxiliary_main_figure_data_extraction/extract_processdata_by_rows.py` is an auxiliary script used to extract selected process-data rows for main-figure source data.

### 02_Model_experiment_codes

Scripts for running the model experiments. The public code keeps the single-pulse and combinational-pulse workflows separate. Each workflow contains one model implementation file, one experiment-matrix file, and one Slurm run script.

```text
single_feature_experiments
|-- model.py
|-- experiment.py
`-- run.sbatch
```

`single_feature_experiments` runs single-pulse feature model evaluations.

```text
combinational_feature_experiments
|-- model.py
|-- experiment.py
`-- run.sbatch
```

`combinational_feature_experiments` runs combinational-pulse model evaluations across three grouped design spaces:

```text
SOC x pulse width, grouped by C-rate
SOC x C-rate, grouped by pulse width
C-rate x pulse width, grouped by SOC
```

The model experiments evaluate capacity/SOH diagnosis using pulse-derived features. The full study uses repeated random splits and reports error metrics such as MAE, RMSE, MAPE, training time, prediction time, and model ranking statistics. These scripts correspond to `05_Model_Experiment_Results` in the associated dataset.

### 03_Main_figure_plotting_codes

Scripts for generating the main manuscript figures.

```text
PulseBat_combo_main_figures
|-- Figure1
|-- Figure2
|-- Figure3
|-- Figure4
`-- Figure5
```

These scripts use the main-figure source data and selected model-result summaries to reproduce the figure panels reported in the manuscript. Supplementary-material plotting scripts and temporary local helper scripts are not included in this repository.

## Data

The associated dataset is deposited separately on Zenodo. It contains:

```text
01_RawData_LFP
02_Preprocessed_step1_workstep_sheet
03_Preprocessed_step2_feature_extraction
04_Model_Input_Feature_Tables
05_Model_Experiment_Results
06_Main_Figure_Source_Data
07_Supplementary_Figure_Source_Data
```

The model-result folder in the dataset contains representative per-case result examples for each experiment type and the complete aggregated analysis results. Full per-case model outputs are not included because of their large size.

## Software Environment

The scripts are written primarily in Python and use standard scientific-computing packages, including:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

The core Python dependencies are listed in `requirements.txt`. Some model-experiment scripts also use additional machine-learning packages depending on the selected model configuration, such as tree-based, Gaussian-process, neural-network, Transformer, or Informer models. The batch scripts are written for a Slurm-based high-performance computing environment and may need path and environment adjustments before reuse.

## License

The code in this repository is released under the MIT License.

## Citation

If you use this code or the associated dataset, please cite the corresponding manuscript once available.
