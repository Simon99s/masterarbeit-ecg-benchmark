# ECG Corruption Benchmark for Robustness Evaluation

This repository contains the code for my master's thesis on evaluating the robustness of ECG classification models under signal corruptions.

The project builds a corrupted ECG benchmark based on the PhysioNet/Computing in Cardiology Challenge 2021 training data, fine-tunes selected ECG classification models, runs inference on clean and corrupted data, evaluates the predictions and generates the thesis plots.

# Project Structure

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

Installation

The code was developed with Python 3.10 Create a new environment and install the required packages:

conda create -n ecg-benchmark python=3.10
conda activate ecg-benchmark
pip install -r requirements.txt
Data

The raw datasets are not included in this repository.

Place the PhysioNet/Computing in Cardiology Challenge 2021 training data in: 

data/raw/physionet.org

Place the MIT-BIH Noise Stress Test Database in:

data/mit-bih-noise-stress-test-database-1.0.0

The generated benchmark will be written to:

data/Benchmark/

The expected benchmark structure after generation is:

data/Benchmark/
├── physionet_ma/
├── physionet_em/
├── physionet_gn/
├── physionet_dn/
└── physionet_in/

Running the Full Pipeline
python runs/01_generate_full_benchmark.py
python runs/02_train_all_models.py
python runs/03_run_full_inference.py
python runs/04_evaluate_all.py
python runs/05_generate_all_plots.py



Individual scripts can also be run directly, for example:

python benchmark_generation/bm_generator.py --artifact ma --severity-name Sev1 --severity-value 7
python benchmark_generation/bm_generator_simulated.py --artifact gn --severity-name Sev1 --severity-value 1
python evaluation/evaluation.py --model xecg