# Fourier Neural Operator (FNO)

This directory contains an implementation of the Fourier Neural Operator (FNO) model for the CTF for Science Framework. The FNO is a deep learning architecture designed for learning operators between function spaces, particularly effective for solving partial differential equations.

## Dependencies

The FNO implementation requires the following dependencies:
- PyTorch (>= 1.8.0)
- NumPy
- CUDA (optional, for GPU acceleration)

## Model Architecture

The FNO model consists of:
1. A Fourier layer that performs spectral convolution in the frequency domain
2. Multiple layers of spectral convolutions and pointwise convolutions
3. A final projection layer to map to the output space

Key components:
- `SpectralConv2d`: Implements the Fourier layer with learnable weights in the frequency domain
- `FNO2d`: Main model architecture combining multiple spectral and pointwise convolutions
- `FNO`: High-level wrapper class that handles data preparation, training, and prediction

## Usage

1. Configure your run by editing the YAML configuration file in `config/`:
   ```yaml
   dataset:
     name: ODE_Lorenz
     pair_id: '1-6'
   model:
     modes1: 12
     modes2: 12
     width: 20
     # ... other parameters
   ```

2. Run the model:
   ```bash
   python run.py config/config_Lorenz.yaml
   ```

## Configuration Parameters

- `modes1`, `modes2`: Number of Fourier modes to use in each dimension
- `width`: Width of the network (number of channels)
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `learning_rate`: Learning rate for training
- `epochs`: Number of training epochs
- `scheduler`: Learning rate scheduler parameters
  - `step_size`: Step size for learning rate decay
  - `gamma`: Multiplicative factor for learning rate decay

## Output

The model generates:
- Predictions for each sub-dataset
- Evaluation metrics (saved in YAML format)
- Visualization plots for trajectories, histograms, and other metrics
- Batch results summary

All outputs are saved in the `results/` directory under a unique batch identifier.

## References

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. arXiv preprint arXiv:2010.08895. 