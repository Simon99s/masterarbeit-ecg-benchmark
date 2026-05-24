# ECG Corruption Benchmark for Robustness Evaluation

This repository contains the code for my master's thesis on evaluating the robustness of ECG classification models under signal corruptions.

The project builds a corrupted ECG benchmark based on the PhysioNet/Computing in Cardiology Challenge 2021 training data, fine-tunes selected ECG classification models, runs inference on clean and corrupted data, evaluates the predictions, and generates the thesis plots.

## Project Structure

```text
masterarbeit-ecg-benchmark/
│
├── benchmark_generation/        # Benchmark generation and verification scripts
├── training/                    # Model-specific fine-tuning scripts
├── inference/                   # Model-specific inference scripts and outputs
├── evaluation/                  # Evaluation code, scoring resources, and result files
├── plotting/                    # Plotting scripts and generated figures
├── data/                        # Raw datasets and generated benchmark data
│
└── runs/                        # Pipeline scripts
    ├── 01_generate_full_benchmark.py
    ├── 02_train_all_models.py
    ├── 03_run_full_inference.py
    ├── 04_evaluate_all.py
    └── 05_generate_all_plots.py
```

## Installation

The code was developed with Python 3.10. Create a new environment and install the required packages:

```bash
conda create -n ecg-benchmark python=3.10
conda activate ecg-benchmark
pip install -r requirements.txt
```

## Data

The raw datasets and generated benchmark files are not included in this repository.

Place the PhysioNet/Computing in Cardiology Challenge 2021 training data in:

```text
data/raw/physionet.org/
```

The scripts expect the training records under:

```text
data/raw/physionet.org/files/challenge-2021/1.0.3/training/
```

Place the MIT-BIH Noise Stress Test Database in:

```text
data/mit-bih-noise-stress-test-database-1.0.0/
```

Place the Subset of the HEEDB in:

```text
data\heedb_subset\heedb_i0006_100.h5/
```

The generated benchmark will be written to:

```text
data/Benchmark/
```

The expected benchmark structure after generation is:

```text
data/Benchmark/
├── physionet_ma/
├── physionet_em/
├── physionet_gn/
├── physionet_dn/
└── physionet_in/
```

## Files Not Included in This Repository

Large files are not stored in this GitHub repository. This includes raw datasets, generated benchmark data, preprocessed HDF5 files, model checkpoints, pretrained weights, inference outputs, and generated evaluation outputs.

These files are stored on the KIS\*MED NAS server in the corresponding project folder.

The following files and folders are expected locally but are not tracked by Git:

```text
data/raw/physionet.org/
data/mit-bih-noise-stress-test-database-1.0.0/
data/Benchmark/

training/**/*.h5
training/**/*.pt
training/**/*.pth
training/**/*.ckpt
training/**/*.safetensors
training/**/*.npy

inference/**/*.pt
inference/**/*.pth
inference/**/*.ckpt
inference/**/*.safetensors
inference/**/*.npy
inference/**/*.npz
inference/**/outputs_2021/
inference/**/tmp_outputs/

evaluation/benchmark_scores.csv
evaluation/feature_collapse_metrics.csv
evaluation/record_ids.npy

Plots/
```

To reproduce the project, copy the required files from the NAS server into the same relative paths in the repository.

## Running the Full Pipeline

```bash
python runs/01_generate_full_benchmark.py
python runs/02_train_all_models.py
python runs/03_run_full_inference.py
python runs/04_evaluate_all.py
python runs/05_generate_all_plots.py
```

## Running Individual Scripts

Individual scripts can also be run directly, for example:

```bash
python benchmark_generation/bm_generator.py --artifact ma --severity-name Sev1 --severity-value 7
python benchmark_generation/bm_generator_simulated.py --artifact gn --severity-name Sev1 --severity-value 1
python evaluation/evaluation.py --model xecg
```