import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        # Compute mean and std along all dimensions except the last one
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x):
        # Ensure x has the same number of dimensions as mean and std
        if x.dim() < self.mean.dim():
            x = x.unsqueeze(0)
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # Ensure x has the same number of dimensions as mean and std
        if x.dim() < mean.dim():
            x = x.unsqueeze(0)
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class SpectralConv2d(nn.Module):
    """
    Spectral Convolution Layer for the Fourier Neural Operator.
    
    This layer performs convolution in the Fourier domain by:
    1. Computing the Fourier coefficients of the input
    2. Multiplying with learnable weights in the frequency domain
    3. Transforming back to physical space
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes in first dimension
        modes2 (int): Number of Fourier modes in second dimension
    
    Raises:
        ValueError: If modes1 or modes2 are negative
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        if modes1 < 0 or modes2 < 0:
            raise ValueError("Number of modes must be non-negative")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 
                                                           self.modes1, self.modes2, 
                                                           dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 
                                                           self.modes1, self.modes2, 
                                                           dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spectral convolution layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
            
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Ensure modes don't exceed input dimensions
        modes1 = min(self.modes1, size1)
        modes2 = min(self.modes2, size2)

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        # Handle first half of modes
        out_ft[:, :, :modes1, :modes2] = \
            torch.einsum("bixy,ioxy->boxy", 
                        x_ft[:, :, :modes1, :modes2], 
                        self.weights1[:, :, :modes1, :modes2])
        
        # Handle second half of modes
        out_ft[:, :, -modes1:, :modes2] = \
            torch.einsum("bixy,ioxy->boxy", 
                        x_ft[:, :, -modes1:, :modes2], 
                        self.weights2[:, :, :modes1, :modes2])

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x

class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator model.
    
    This model combines multiple spectral convolution layers with pointwise convolutions
    to learn operators between function spaces.
    
    Args:
        modes1 (int): Number of Fourier modes in first dimension
        modes2 (int): Number of Fourier modes in second dimension
        width (int): Width of the network
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    Raises:
        ValueError: If any of the input parameters are invalid
    """
    def __init__(self, modes1: int, modes2: int, width: int, 
                 in_channels: int, out_channels: int):
        super(FNO2d, self).__init__()
        
        if modes1 <= 0 or modes2 <= 0:
            raise ValueError("Number of modes must be positive")
        if width <= 0:
            raise ValueError("Network width must be positive")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Number of channels must be positive")

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FNO model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")

        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class FNO:
    """
    Fourier Neural Operator (FNO) model wrapper.
    
    This class provides a high-level interface for training and using the FNO model,
    handling data preparation, model initialization, training, and prediction.
    
    Attributes:
        pair_id (int): Identifier for the data pair to consider
        train_data (np.ndarray): Training data
        m (int): Number of time points
        n (int): Number of spatial points
        init_data (np.ndarray): Burn-in data for prediction
        prediction_horizon_steps (int): Number of timesteps to predict
        modes1 (int): Number of Fourier modes in first dimension
        modes2 (int): Number of Fourier modes in second dimension
        width (int): Width of the network
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        device (torch.device): Device to run the model on
        model (FNO2d): The FNO model
        config (Dict): Configuration dictionary
    """
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, 
                 init_data: Optional[np.ndarray] = None, 
                 prediction_horizon_steps: int = 0, 
                 pair_id: Optional[int] = None):
        """
        Initialize the FNO model with the provided configuration.

        Args:
            config (Dict): Configuration dictionary containing method and parameters
            train_data (Optional[np.ndarray]): Training data for the model
            init_data (Optional[np.ndarray]): Initialization data for prediction
            prediction_horizon_steps (int): Number of timesteps to predict
            pair_id (Optional[int]): Identifier for the data pair to consider
        """
        # Store config
        self.config = config
        
        # Validate configuration
        required_params = ['modes1', 'modes2', 'width', 'in_channels', 'out_channels']
        for param in required_params:
            if param not in config['model']:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu' and torch.cuda.is_available():
            logger.warning("CUDA is available but not being used")
        self.dtype = torch.float32

        # Store data
        self.pair_id = pair_id
        self.train_data = train_data
        if train_data is not None:
            self.m = train_data.shape[0]  # Number of time points
            self.n = train_data.shape[1]  # Number of spatial points
            logger.debug(f"Training data shape: {train_data.shape}")
            logger.debug(f"Number of time points (m): {self.m}")
            logger.debug(f"Number of spatial points (n): {self.n}")
        self.init_data = init_data
        if init_data is not None:
            logger.debug(f"Initialization data shape: {init_data.shape}")
        self.prediction_horizon_steps = prediction_horizon_steps
        logger.debug(f"Prediction horizon steps: {prediction_horizon_steps}")

        # Initialize normalizer
        if train_data is not None:
            # Normalize each spatial dimension separately
            self.normalizers = []
            for i in range(self.n):
                logger.debug(f"Creating normalizer for spatial dimension {i}")
                logger.debug(f"Data shape for dimension {i}: {train_data[:, i].shape}")
                normalizer = UnitGaussianNormalizer(torch.from_numpy(train_data[:, i]).type(self.dtype))
                if self.device.type == 'cuda':
                    normalizer.cuda()
                self.normalizers.append(normalizer)

        # Model parameters
        self.modes1 = config['model']['modes1']
        self.modes2 = config['model']['modes2']
        self.width = config['model']['width']
        self.in_channels = config['model']['in_channels']
        self.out_channels = config['model']['out_channels']
        logger.debug(f"Model parameters: modes1={self.modes1}, modes2={self.modes2}, width={self.width}")
        logger.debug(f"Channels: in={self.in_channels}, out={self.out_channels}")

        # Initialize model
        self.model = FNO2d(self.modes1, self.modes2, self.width, 
                          self.in_channels, self.out_channels).to(self.device)

    def get_data(self) -> Dict[str, torch.Tensor]:
        """
        Prepare the training data for the model.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing training data
            
        Raises:
            ValueError: If training data is not available
        """
        if self.train_data is None:
            raise ValueError("Training data is required but not provided")
            
        # Convert data to torch tensors
        train_data = torch.from_numpy(self.train_data).type(self.dtype).to(self.device)
        logger.debug(f"Training data tensor shape: {train_data.shape}")
        
        # Normalize each spatial dimension separately
        normalized_data = torch.zeros_like(train_data)
        for i in range(self.n):
            logger.debug(f"Normalizing spatial dimension {i}")
            logger.debug(f"Input shape for dimension {i}: {train_data[:, i].shape}")
            normalized_data[:, i] = self.normalizers[i].encode(train_data[:, i])
            logger.debug(f"Output shape for dimension {i}: {normalized_data[:, i].shape}")
        
        # Reshape data for FNO model
        # Input shape: [batch_size, channels, height, width]
        # For 1D data, we'll use width=1 and height=spatial_dim
        x = normalized_data[:-1].unsqueeze(1).unsqueeze(-1)  # [batch, channels, height, width]
        y = normalized_data[1:].unsqueeze(1).unsqueeze(-1)   # [batch, channels, height, width]
        logger.debug(f"Input shape after reshaping: {x.shape}")
        logger.debug(f"Output shape after reshaping: {y.shape}")

        return {
            "train_input": x,
            "train_label": y
        }

    def train(self) -> None:
        """
        Train the FNO model.
        
        Raises:
            ValueError: If training data is not available
        """
        data = self.get_data()
        
        # Get training parameters from config
        learning_rate = float(self.config['model'].get('learning_rate', 1e-3))
        epochs = int(self.config['model'].get('epochs', 500))
        batch_size = int(self.config['model'].get('batch_size', 32))
        optimizer_name = self.config['model'].get('optimizer', 'adam').lower()
        activation_name = self.config['model'].get('activation', 'gelu').lower()
        
        # Set activation function
        if activation_name == 'gelu':
            activation = F.gelu
        elif activation_name == 'relu':
            activation = F.relu
        elif activation_name == 'tanh':
            activation = torch.tanh
        else:
            logger.warning(f"Unknown activation function {activation_name}, using GELU")
            activation = F.gelu
        
        # Initialize optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize scheduler
        scheduler_config = self.config['model'].get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'StepLR')
        if scheduler_name == 'StepLR':
            step_size = int(scheduler_config.get('step_size', 100))
            gamma = float(scheduler_config.get('gamma', 0.5))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            logger.warning(f"Unknown scheduler {scheduler_name}, using StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        # Training loop
        n_samples = data["train_input"].shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                # Get batch data
                batch_input = data["train_input"][start_idx:end_idx]
                batch_label = data["train_label"][start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.model(batch_input)
                loss = F.mse_loss(y_pred, batch_label)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Log progress
            if epoch % 100 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
                logger.debug(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')

    def predict(self) -> np.ndarray:
        """
        Generate predictions using the trained model.
        """
        if self.init_data is None:
            if self.train_data is None:
                raise ValueError("Neither initialization data nor training data is available")
            # Use the last timestep of training data as initialization
            self.init_data = self.train_data[-1:]
            logger.debug("Using last timestep of training data as initialization")
            
        self.model.eval()
        with torch.no_grad():
            # Use init_data for prediction
            x = torch.from_numpy(self.init_data).type(self.dtype).to(self.device)
            logger.debug(f"Initial data shape: {x.shape}")
            
            # Normalize each spatial dimension separately
            normalized_data = torch.zeros_like(x)
            for i in range(self.n):
                logger.debug(f"Normalizing spatial dimension {i}")
                logger.debug(f"Input shape for dimension {i}: {x[:, i].shape}")
                normalized_data[:, i] = self.normalizers[i].encode(x[:, i])
                logger.debug(f"Output shape for dimension {i}: {normalized_data[:, i].shape}")
            
            x = normalized_data.unsqueeze(1).unsqueeze(-1)  # Add channel and width dimensions
            logger.debug(f"Input shape after reshaping: {x.shape}")
            
            predictions = []
            for step in range(self.prediction_horizon_steps):
                y = self.model(x)
                logger.debug(f"Step {step} prediction shape: {y.shape}")
                predictions.append(y)
                x = y  # Use prediction as next input
            
            # Stack predictions along time dimension and remove extra dimensions
            predictions = torch.stack(predictions, dim=0)  # [time_steps, batch, channels, height, width]
            predictions = predictions.squeeze(1).squeeze(-1)  # Remove batch and width dimensions
            logger.debug(f"Final predictions shape before denormalization: {predictions.shape}")
            
            # Reshape predictions to [time_steps, spatial_dim]
            predictions = predictions.squeeze(1)  # Remove channel dimension
            logger.debug(f"Predictions shape after squeezing: {predictions.shape}")
            
            # Initialize denormalized predictions with correct shape
            denormalized_predictions = torch.zeros((predictions.shape[0], self.n), 
                                                 dtype=self.dtype, 
                                                 device=self.device)
            
            # Denormalize each spatial dimension separately
            for i in range(self.n):
                logger.debug(f"Denormalizing spatial dimension {i}")
                logger.debug(f"Input shape for dimension {i}: {predictions[:, i].shape}")
                denormalized_predictions[:, i] = self.normalizers[i].decode(predictions[:, i])
                logger.debug(f"Output shape for dimension {i}: {denormalized_predictions[:, i].shape}")
            
        return denormalized_predictions.cpu().numpy() 