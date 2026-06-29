# Pulse Excitation Informativeness for Battery Capacity Diagnosis

This repository contains the code and lightweight processed data accompanying the study:

**Constant-flux diffusion scaling maps pulse-excitation informativeness for battery capacity diagnosis**

The study maps how the diagnostic informativeness of pulse excitation depends on pulse current intensity, pulse duration, and charge status for battery capacity diagnosis. We use lithium iron phosphate (LFP) cells with nominal capacities of 20 Ah, 35 Ah, and 68 Ah, and evaluate both single-pulse excitation and combinational-pulse excitation protocols. Diagnosis error is used as the measure of pulse-excitation informativeness, and a diffusion-scaled descriptor derived from constant-flux diffusion scaling is used to organize the observed error landscapes.

This GitHub repository is the main code repository. It contains the analysis code, the final model-input feature tables, representative model-result examples, and main-figure source data. The large raw experimental data are distributed separately through Zenodo.

## Zenodo Raw Data

Download the raw LFP pulse-excitation data from Zenodo:

```text
Zenodo raw-data link: <insert Zenodo DOI or URL here>
```

After downloading, place the raw data under this repository as:

```text
Data
`-- 01_RawData_LFP
    |-- 20Ah LFP
    |-- 35Ah LFP
    `-- 68Ah LFP
```

The expected local path after download is:

```text
./Data/01_RawData_LFP
```

The raw-data archive corresponds to:

```text
G:\Paper-Mapping_informativeness_of_pulse_excitation_for_scalable_battery_capacity_diagnosis_through_diffusion-scaled_electrochemical_perturbations\Data\01_RawData_LFP
```

Intermediate step-1 and step-2 processed data are not stored in this GitHub repository. They can be regenerated from the Zenodo raw data using the preprocessing scripts below. The repository already includes the final model-input feature tables for direct model reuse.

## Repository Structure

```text
.
|-- 01_Data_preprocessing_codes_with_Data
|   |-- Codes_PulseBat_raw_to_model_features
|   |   |-- step_1_extract workstep sheet.py
|   |   |-- step_2_feature extraction_adjustable.py
|   |   |-- step_2_feature extraction_complete.py
|   |   |-- step_3_feature collection_adjustable.py
|   |   `-- step_3_feature collection_complete.py
|   `-- Data_Model_Input_Feature_Tables
|       |-- 20Ah LFP/*.xlsx
|       |-- 35Ah LFP/*.xlsx
|       `-- 68Ah LFP/*.xlsx
|-- 02_Model_Experiment_Codes_with_Data
|   |-- single_feature_experiments
|   |   |-- model.py
|   |   |-- experiment.py
|   |   `-- run.sbatch
|   |-- Codes-Combinational_Feature_Experiments
|   |   |-- model.py
|   |   |-- experiment.py
|   |   `-- run.sbatch
|   `-- Data-Model_Experiment_Results_Examples
|       |-- single_feature_results
|       |-- combo_exp1_soc_width_results
|       |-- combo_exp2_soc_crate_results
|       `-- combo_exp3_crate_width_results
|-- 03_Main_Figure_Plotting_Codes_with_Data
|   |-- Figure1
|   |-- Figure2
|   |-- Figure3
|   |-- Figure4
|   `-- Figure5
|-- Data
|   `-- 01_RawData_LFP
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Reproducible Workflow

The intended workflow is:

```text
Zenodo raw data
-> preprocessing step 1
-> preprocessing step 2
-> preprocessing step 3
-> model-input feature tables
-> single-feature and combinational-feature model experiments
-> model-result examples / regenerated full results
-> main-figure plotting
```

### 1. Install Dependencies

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Some deep-learning models require `torch`. Some tree-based model configurations require `xgboost`. The Slurm scripts are provided for high-performance computing environments and may need local cluster-specific changes.

### 2. Download Raw Data

Download the raw LFP data from Zenodo and place it at:

```text
./Data/01_RawData_LFP
```

Expected subfolders:

```text
./Data/01_RawData_LFP/20Ah LFP
./Data/01_RawData_LFP/35Ah LFP
./Data/01_RawData_LFP/68Ah LFP
```

### 3. Run Preprocessing

Preprocessing code is in:

```text
01_Data_preprocessing_codes_with_Data/Codes_PulseBat_raw_to_model_features
```

Run the scripts in order:

```bash
cd 01_Data_preprocessing_codes_with_Data/Codes_PulseBat_raw_to_model_features
python "step_1_extract workstep sheet.py"
python "step_2_feature extraction_adjustable.py"
python "step_2_feature extraction_complete.py"
python "step_3_feature collection_adjustable.py"
python "step_3_feature collection_complete.py"
```

The preprocessing logic is:

```text
step 1: raw Excel files -> workstep-level records
step 2: workstep-level records -> pulse-response feature extraction
step 3: extracted pulse features -> final model-input feature tables
```

The final feature tables used by the model experiments are included in:

```text
01_Data_preprocessing_codes_with_Data/Data_Model_Input_Feature_Tables
```

These included feature tables are the practical entry point for reproducing the model experiments without regenerating the intermediate step-1 and step-2 processed data. The retained model-input columns include capacity/SOH metadata, the voltage response vector `U1` to `U41`, and the pulse feature columns `Fai_0.5C`, `Fai_1C`, `Fai_1.5C`, `Fai_2C`, and `Fai_2.5C`. `Fai_*C` is the ASCII table label for the pulse feature denoted as `phi(C)` in the manuscript and Supplementary Note 3. If rerunning preprocessing from raw data, update the input/output paths in the preprocessing scripts to point to `./Data/01_RawData_LFP` and to the desired local output folders.

### 4. Run Model Experiments

Model experiment code is in:

```text
02_Model_Experiment_Codes_with_Data
```

The repository includes only representative model-result examples in:

```text
02_Model_Experiment_Codes_with_Data/Data-Model_Experiment_Results_Examples
```

These examples show the output format for one selected case from each experiment type. They are not the full experimental output. To reproduce the full work, run the provided model experiment scripts using the model-input feature tables.

#### Single-Feature Experiments

Code:

```text
02_Model_Experiment_Codes_with_Data/single_feature_experiments
```

PowerShell example:

```powershell
cd 02_Model_Experiment_Codes_with_Data\single_feature_experiments
$env:DATA_ROOT = "..\..\01_Data_preprocessing_codes_with_Data\Data_Model_Input_Feature_Tables"
$env:RESULT_BASE_DIR = "..\Reproduced_Model_Results"
python experiment.py
```

Slurm example:

```bash
cd 02_Model_Experiment_Codes_with_Data/single_feature_experiments
export DATA_ROOT="../../01_Data_preprocessing_codes_with_Data/Data_Model_Input_Feature_Tables"
export RESULT_BASE_DIR="../Reproduced_Model_Results"
sbatch run.sbatch
```

The single-feature experiment evaluates individual `fai_irrev`-style pulse features across LFP material size, SOC, and pulse-width conditions.

#### Combinational-Feature Experiments

Code:

```text
02_Model_Experiment_Codes_with_Data/Codes-Combinational_Feature_Experiments
```

PowerShell example:

```powershell
cd 02_Model_Experiment_Codes_with_Data\Codes-Combinational_Feature_Experiments
$env:DATA_ROOT = "..\..\01_Data_preprocessing_codes_with_Data\Data_Model_Input_Feature_Tables"
$env:RESULT_BASE_DIR = "..\Reproduced_Model_Results"
$env:EXPERIMENTS = "soc_width,soc_crate,crate_width"
python experiment.py
```

Slurm example:

```bash
cd 02_Model_Experiment_Codes_with_Data/Codes-Combinational_Feature_Experiments
export DATA_ROOT="../../01_Data_preprocessing_codes_with_Data/Data_Model_Input_Feature_Tables"
export RESULT_BASE_DIR="../Reproduced_Model_Results"
export EXPERIMENTS="soc_width,soc_crate,crate_width"
sbatch run.sbatch
```

The combinational-feature experiments cover three grouped design spaces:

```text
soc_width   : SOC x pulse width, grouped by C-rate
soc_crate   : SOC x C-rate, grouped by pulse width
crate_width : C-rate x pulse width, grouped by SOC
```

The full experiment matrix is computationally large. The included `Data-Model_Experiment_Results_Examples` folder is provided only as a compact example of result structure. Full reproduction requires running the experiment scripts on the complete model-input feature tables.

### 5. Plot Main Figures

Main-figure plotting code and source data are in:

```text
03_Main_Figure_Plotting_Codes_with_Data
```

Each figure folder contains the plotting scripts and the CSV/XLSX source data needed by that figure. Example:

```bash
cd 03_Main_Figure_Plotting_Codes_with_Data/Figure2/Figure2a
python Figure2a.py
```

Figure 5 can be run panel-by-panel or with:

```bash
cd 03_Main_Figure_Plotting_Codes_with_Data/Figure5
python run_all_figure5.py
```

`Figure1/Figure1c/extract_processdata_by_rows.py` is a figure-specific helper for extracting Figure 1c process-data snippets from raw Excel files. It is not part of the general model preprocessing pipeline.

Some plotting scripts still contain local path defaults from the original analysis environment. When rerunning outside that environment, pass available command-line arguments where provided or update the path constants at the top of the relevant figure script to point to the source data included in this repository.

## Folder Purposes

### `01_Data_preprocessing_codes_with_Data`

Contains the raw-data preprocessing scripts and the final model-input feature tables. The included `Data_Model_Input_Feature_Tables` folder is the direct input to the model experiments.

### `02_Model_Experiment_Codes_with_Data`

Contains model training/evaluation code and representative model-result examples. The examples are included to document the output format. The complete result set can be regenerated by running the single-feature and combinational-feature experiment scripts.

### `03_Main_Figure_Plotting_Codes_with_Data`

Contains main-text figure plotting scripts and the corresponding figure source data. The source data are stored next to the figure scripts that use them.

## License

The code in this repository is released under the MIT License.

## Citation

If you use this code or the associated dataset, please cite the corresponding manuscript once available.
