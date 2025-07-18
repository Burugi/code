# Dissecting Channel Dependency in Multivariate Time Series Forecasting: Interaction Scope, Level, and Architecture

This project provides experimental code for **Multivariate Time Series Forecasting**, comparing over 18 models across 10 datasets to analyze performance differences between **Channel-Dependent (CD)** and **Channel-Independent (CI)** forecasting approaches. It also includes a new dataset called Milano.

## 📁 Project Structure

- `run_hyperopt.py`: Hyperparameter optimization
- `run_training.py`: Model training

## 📦 Environment Setup

1. **Set up Python environment and install packages**

```bash
conda create -n CICD_compairson python=3.8
conda activate CICD_compairson
pip install -r requirements.txt
```

2. **Prepare Dataset**

- Create a `dataset/` folder and place CSV datasets used in the 'Are Transformers Effective for Time Series Forecasting?' paper.
- Include the new Milano dataset as `Milano.csv`.

## 🚀 How to Run

### 1. Hyperparameter Optimization

```bash
python run_hyperopt.py \
  --file_name Milano.csv \
  --models dlinear tcn transformer \
  --modes CD CI \
  --target_feature OT \
  --n_trials 10
```

- `--models`: List of target models
- `--modes`: Experiment modes (`CD`, `CI`)
- `--target_feature`: Feature to predict in CI mode
- `--n_trials`: Number of Optuna optimization trials

### 2. Model Training

```bash
python run_training.py \
  --file_name Milano.csv \
  --models dlinear tcn transformer \
  --modes CD CI \
  --target_feature OT \
  --epochs 50 \
  --n_repeats 3
```

- Trained results are saved under `training_results/`.

### 3. Visualization

```bash
python run_visualization.py \
  --file_name Milano.csv \
  --models dlinear tcn transformer \
  --modes CD CI \
  --target_feature OT
```

- Visual results are saved in `visualizations/`.
- Visualizes:
  - Training history
  - Feature-wise metrics
  - Model performance comparisons
  - CD vs CI prediction differences

## 📊 Included Models

- DLinear (+ Linear)
- TCN
- Transformer
- RNN
- Autoformer
- TimeMixer
- TSMixer
- SegRNN
- SCINet
- TimesNet
- TiDE
- LightTS
- Pyraformer
- Informer
- DSSRNN (+ SSRNN)
- MICN

## 📂 Example Directory Layout

```
.
├── dataset/
│   └── Milano.csv
├── training_results/
│   └── dlinear/
│       └── CD/
│       └── CI/OT/
├── visualizations/
│   └── comparisons/
├── configs/
│   └── hyperopt_config.yaml
├── run_training.py
├── run_hyperopt.py
├── run_visualization.py
└── requirements.txt
```

## 📌 Notes

- All models follow preprocessing defined in `data/dataloader.py`.
- CD mode predicts all features simultaneously, while CI mode focuses on one specified feature.
- Experiment repetition and early stopping parameters can be customized via command-line arguments.
