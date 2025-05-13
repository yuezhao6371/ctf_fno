# Fourier Neural Operator (FNO)

This directory contains an implementation of the Fourier Neural Operator (FNO) model for the CTF for Science Framework. The FNO is a deep learning architecture designed for learning operators between function spaces, particularly effective for solving partial differential equations.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the model, use the `run.py` script from the model directory:

```bash
cd models/ctf_fno
python run.py config/config_Lorenz.yaml
python run.py config/config_KS.yaml
```

### Hyperparameter Tuning

To run hyperparameter tuning:

```bash
cd models/ctf_fno
python optimize_parameters.py --metric 'score' --mode 'max'
```

## Configuration Files

Configuration files are located in the `config/` directory:
- `config_KS.yaml`: Runs the FNO model on `PDE_KS` for all sub-datasets
- `config_Lorenz.yaml`: Runs the FNO model on `ODE_Lorenz` for all sub-datasets

## Dependencies

The FNO implementation requires the following dependencies:
- PyTorch (>= 1.8.0)
- NumPy (>= 1.19.0)
- PyYAML (>= 5.1.0)
- Ray [tune] (>= 2.0.0) for hyperparameter tuning

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

## Output

The model generates:
- Predictions for each sub-dataset
- Evaluation metrics (saved in YAML format)
- Visualization plots for trajectories, histograms, and other metrics
- Batch results summary

All outputs are saved in the `results/` directory under a unique batch identifier.

## References

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. arXiv preprint arXiv:2010.08895. 
