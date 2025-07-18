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


## 📎 Appendix: Hyperparameter Search Spaces

This section outlines the comprehensive hyperparameter search ranges used for each of the 15 models evaluated in this study. The ranges were derived from preliminary experiments and prior literature to ensure fair and consistent comparison across all models.

### 🔧 Categories of Hyperparameters

- **Common Parameters**: Learning rate, batch size, dropout rate, number of epochs  
- **Architecture-Specific Parameters**:  
  - Hidden dimensions, number of layers, attention heads (for Transformers), kernel sizes (for CNNs)  
- **Optimization Settings**: Optimizer type, weight decay, learning rate scheduling  
- **Model-Specific Hyperparameters**:  
  - ProbSparse attention (Informer), decomposition settings (Autoformer), and more

### 📐 Hyperparameter Search Ranges

| Hyperparameter          | Search Range                      |
|-------------------------|------------------------------------|
| `learning_rate`         | [0.0001, 0.01]                     |
| `batch_size`            | [32, 64, 128, 256]                |
| `dropout`               | [0.1, 0.5]                         |
| `weight_decay`          | [0.000001, 0.01]                   |
| `d_model`               | [64, 128, 256, 512]                |
| `n_heads`               | [4, 8, 16]                         |
| `e_layers`, `d_layers`  | [1, 2, 3, 4]                       |
| `d_ff`                  | [128, 256, 512, 1024]              |
| `moving_avg`            | [13, 25, 37]                       |
| `factor`                | [1, 2, 3, 4, 5]                    |
| `embed`                 | `{fixed, timeF}`                   |
| `activation`            | `{relu, gelu}`                     |
| `kernel_size`           | [3, 45] (varies by model)          |
| `conv_kernel`           | {[12,16], [8,12], [16,24]}         |
| `window_size`           | {[2,2], [4,4], [8,4]}              |
| `inner_size`            | [3, 7]                             |
| `hidden_size`           | [64, 128, 256, 512]                |
| `num_layers`            | [1, 2, 3, 4]                       |
| `label_len`             | [0, 48]                            |
| `distil`                | `{true, false}`                    |
| `chunk_size`            | [4, 36]                            |
| `rnn_type`              | `{LSTM, GRU}`                      |
| `levels`                | [2, 4]                             |
| `stacks`                | [1, 2]                             |
| `seg_len`               | [6, 12, 24]                        |
| `feature_encode_dim`    | [2, 8]                             |
| `down_sampling_layers` | [1, 3]                             |
| `down_sampling_window` | [2, 4]                             |
| `decomp_method`         | `{moving_avg, dft}`                |
| `top_k`                 | [1, 7]                             |
| `num_kernels`           | [4, 8]                             |


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
