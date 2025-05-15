# Fourier Neural Operator (FNO)

This directory contains an implementation of the Fourier Neural Operator (FNO) model for the CTF for Science Framework. The FNO is a deep learning architecture designed for learning operators between function spaces.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Output](#output)
- [References](#references)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -U pip
pip install -r requirements.txt
cd ../..
pip install -e .[all]
```

## Usage

### Running the Model

To run the model, use the `run.py` script from the model directory:

```bash
cd models/ctf_fno
python run.py config/config_Lorenz_batch.yaml
python run.py config/config_KS_batch.yaml
```

### Hyperparameter Tuning

To run hyperparameter tuning:

```bash
cd models/ctf_fno
python optimize_parameters.py --metric 'score' --mode 'max'
```

Additional tuning options:
- `--time-budget-hours`: Maximum time budget for tuning (default: 24.0)
- `--use-asha`: Enable ASHA scheduler for early stopping
- `--gpus-per-trial`: Number of GPUs per trial (default: 0, meaning use all available)
- `--config-path`: Specify a config path to tune with only that config file (default: all config_*.yaml under tuning_config)
- use n_trials parameter in the config file to limit the number of trials

## Configuration

Configuration files are located in the `config/` directory:
- `config_KS_batch.yaml`: Runs the FNO model on `PDE_KS`
- `config_Lorenz_batch.yaml`: Runs the FNO model on `ODE_Lorenz`

Each configuration file contains:
- Dataset specifications
- Model hyperparameters
- Training parameters

## Dependencies

The FNO implementation requires the following dependencies:
- PyTorch (>= 1.8.0, < 2.0.0)
- NumPy (>= 1.19.0, < 2.0.0)
- PyYAML (>= 5.1.0, < 6.0.0)
- ctf4science python project

See `requirements.txt` for the complete list of dependencies.

## Model Architecture

The FNO model consists of:
1. A Fourier layer that performs spectral convolution in the frequency domain
2. Multiple layers of spectral convolutions and pointwise convolutions
3. A final projection layer to map to the output space

Key components:
- `SpectralConv2d`: Implements the Fourier layer with learnable weights in the frequency domain
- `FNO2d`: Main model architecture combining multiple spectral and pointwise convolutions
- `FNO`: High-level wrapper class that handles data preparation, training, and prediction

### Architecture Details
- Input: Time series data with spatial dimensions
- Fourier Transform: Converts input to frequency domain
- Spectral Convolution: Learns frequency domain features
- Inverse Fourier Transform: Converts back to spatial domain
- Output: Predicted time series

## Output

The model generates several types of outputs:

### Training Outputs from run.py
- Predictions for each sub-dataset
- Evaluation metrics (saved in YAML format)
- Batch results summary
- Location: `results/` directory under a unique batch identifier

### Tuning Outputs from optimize_parameters.py
- Optimal hyperparameters
- Tuning history
- Performance metrics
- Location: `results/tune_result` directory

## References

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. arXiv preprint arXiv:2010.08895. 
