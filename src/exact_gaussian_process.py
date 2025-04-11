from sklearn.compose import ColumnTransformer
import torch
import gpytorch
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        # Mean module - constant mean (instead of zero mean)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Covariance module - RBF (Gaussian) kernel with ARD
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1))
        )
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class ExactGaussianProcess:
    def __init__(self, 
                 non_category_features: list,
                 num_epochs: int = 100, 
                 learning_rate: float = 0.1):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = None
        self.likelihood = None
        self.x_scaler = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), non_category_features)   # StandardScaler()
            ],
            remainder='passthrough'  # Leave categorical features untouched
        )
        self.y_scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale the input features and target
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        return X_tensor, y_tensor
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Process model to the training data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(X_tensor, y_tensor, self.likelihood).to(self.device)
        
        # Set model and likelihood to training mode
        self.model.train()
        self.likelihood.train()
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=self.learning_rate)
        
        # Loss function - negative log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            return_std: If True, return standard deviation of predictions
            
        Returns:
            mean: Mean predictions
            std: Standard deviation of predictions (if return_std=True)
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Scale input features
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_tensor))
            mean = predictions.mean.cpu().numpy()
            
            # Inverse transform predictions
            mean = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
            
            if return_std:
                std = predictions.stddev.cpu().numpy()
                # Scale back the standard deviation
                std = std * self.y_scaler.scale_
                return mean, std
            
            return mean, None

    def get_params(self) -> dict:
        """Get the learned hyperparameters of the model."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        params = {
            'constant_mean': self.model.mean_module.constant.item(),
            'lengthscales': self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
            'outputscale': self.model.covar_module.outputscale.detach().cpu().numpy(),
            'noise': self.likelihood.noise.detach().cpu().numpy()
        }
        return params
