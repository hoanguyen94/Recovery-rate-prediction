import torch
import gpytorch
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.utils.cholesky import psd_safe_cholesky
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class SparsePEPModel(gpytorch.models.ApproximateGP):
    """
    Sparse GP model using Power Expectation Propagation
    """
    def __init__(self, inducing_points, num_features, alpha=0.5):
        # Initialize variational distribution
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # Initialize variational strategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance modules with stability constraints
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF kernel with ARD and stability constraints
        self.rbf_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=num_features,
            lengthscale_constraint=gpytorch.constraints.Interval(1e-6, 1e6)
        )
        
        # Combined kernel with scale
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.rbf_kernel,
            outputscale_constraint=gpytorch.constraints.Interval(1e-6, 1e6)
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class PowerExpectationPropagation(MarginalLogLikelihood):
    """
    Power Expectation Propagation (PEP) implementation for GPyTorch.
    """
    def __init__(self, likelihood, model, num_data, alpha=0.5):
        super().__init__(likelihood, model)
        self.alpha = alpha
        self.num_data = num_data
        self.min_jitter = 1e-8
        self.max_jitter = 1.0
        self.jitter_steps = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]

    def _add_jitter_until_pd(self, matrix):
        """Incrementally add jitter until matrix is positive definite"""
        matrix_size = matrix.size(-1)
        for jitter in self.jitter_steps:
            try:
                adjusted_matrix = matrix + torch.eye(matrix_size, device=matrix.device) * jitter
                L = torch.linalg.cholesky(adjusted_matrix)
                return L, adjusted_matrix, jitter
            except RuntimeError:
                continue
        
        # If all steps fail, return None to handle the error case
        return None, None, None

    def _safe_cholesky(self, matrix):
        """Attempt Cholesky decomposition with increasing jitter and scaling"""
        # Try standard Cholesky first
        L, adjusted_matrix, used_jitter = self._add_jitter_until_pd(matrix)
        if L is not None:
            return L, adjusted_matrix
        
        # If standard approach fails, try scaling
        scale = matrix.abs().max()
        if scale > 1e-8:
            scaled_matrix = matrix / scale
            L, adjusted_matrix, used_jitter = self._add_jitter_until_pd(scaled_matrix)
            if L is not None:
                return L * scale.sqrt(), adjusted_matrix * scale
        
        raise RuntimeError("Failed to compute Cholesky decomposition even with maximum jitter")

    def forward(self, output, target):
        if not isinstance(output, gpytorch.distributions.MultivariateNormal):
            raise RuntimeError("Output must be a MultivariateNormal distribution")

        try:
            # Get the covariance matrices
            Kmm = output.lazy_covariance_matrix.to_dense()
            Knn = output.lazy_covariance_matrix.diagonal()
            
            # Make sure Kmm is symmetric
            Kmm = 0.5 * (Kmm + Kmm.transpose(-1, -2))
            
            # Safe Cholesky decomposition of Kmm
            L, Kmm_adjusted = self._safe_cholesky(Kmm)
            
            # Compute beta_star with stability constraints
            sigma_n = torch.clamp(self.likelihood.noise, min=1e-6, max=1.0)
            diag_term = torch.clamp(torch.sum(torch.square(L), dim=0), min=1e-8)
            sigma_star = sigma_n + self.alpha * (Knn - diag_term)
            sigma_star = torch.clamp(sigma_star, min=1e-6)
            beta_star = 1.0 / sigma_star
            
            # Compute and stabilize matrix A
            weighted_L = L * torch.sqrt(beta_star).unsqueeze(0)
            A = torch.matmul(weighted_L.t(), weighted_L)
            A = A + torch.eye(A.size(0), device=A.device) * 1e-6
            
            # Safe Cholesky decomposition of A
            LA, A_adjusted = self._safe_cholesky(A)
            
            # Compute intermediate terms with numerical safeguards
            URiy = torch.matmul(Kmm_adjusted.t() * beta_star, target)
            
            # Safe triangular solves
            tmp = torch.triangular_solve(URiy.unsqueeze(-1), L, upper=False)[0]
            b = torch.triangular_solve(tmp, LA, upper=False)[0]
            tmp = torch.triangular_solve(b, LA.t(), upper=True)[0]
            v = torch.triangular_solve(tmp, L.t(), upper=True)[0].squeeze(-1)
            
            # Compute log marginal likelihood terms with stability
            alpha_const_term = (1.0 - self.alpha) / self.alpha
            
            log_det_term = -torch.sum(torch.log(torch.diagonal(LA) + 1e-8))
            quad_term = -0.5 * torch.sum(torch.square(target * torch.sqrt(beta_star)))
            trace_term = 0.5 * torch.sum(torch.square(b))
            const_term = 0.5 * alpha_const_term * self.num_data * torch.log(sigma_n)
            beta_term = 0.5 * (1 + alpha_const_term) * torch.sum(torch.log(beta_star))
            
            # Combine terms with stability check
            nll = -(
                log_det_term +
                quad_term +
                trace_term +
                const_term +
                beta_term -
                0.5 * self.num_data * np.log(2 * np.pi)
            )
            
            # Add small regularization term
            nll = nll + 1e-4 * (torch.sum(torch.square(L)) + torch.sum(torch.square(LA)))
            
            # Check for invalid values
            if torch.isnan(nll) or torch.isinf(nll):
                return torch.tensor(float('inf'), device=target.device, requires_grad=True)
                
            return nll
            
        except RuntimeError as e:
            print(f"Numerical error in PEP computation: {str(e)}")
            return torch.tensor(float('inf'), device=target.device, requires_grad=True)


class PEPGaussianProcess:
    """
    Sparse Gaussian Process model using Power Expectation Propagation for inference.
    """
    def __init__(self, non_category_features: list, alpha=0.5, learning_rate=0.01, 
                 num_epochs=100, num_inducing=100):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_inducing = num_inducing
        self.model = None
        
        # Initialize preprocessing components
        self.x_scaler = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), non_category_features)
            ],
            remainder='passthrough'
        )
        self.y_scaler = StandardScaler()
        
        # Initialize likelihood with stability constraints
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.likelihood = self.likelihood.to(self.device)

    def _prepare_data(self, X: np.ndarray, y: np.ndarray):
        """Prepare data by scaling features and target"""
        # Scale the input features and target
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        return X_tensor, y_tensor

    def _select_inducing_points(self, train_x):
        """Select inducing points using k-means"""
        kmeans = KMeans(n_clusters=self.num_inducing)
        kmeans.fit(train_x.cpu().numpy())
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model using PEP with enhanced stability measures"""
        # Prepare data
        train_x, train_y = self._prepare_data(X, y)
        
        # Select inducing points and create model
        inducing_points = self._select_inducing_points(train_x).to(self.device)
        self.model = SparsePEPModel(
            inducing_points=inducing_points,
            num_features=train_x.size(1),
            alpha=self.alpha
        ).to(self.device)
        
        self.model.train()
        self.likelihood.train()
        
        # Initialize PEP loss with stability settings
        mll = PowerExpectationPropagation(
            self.likelihood, 
            self.model,
            num_data=len(train_x),
            alpha=self.alpha
        )
        
        # Optimizer with gradient clipping and smaller learning rate
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.001)  # Reduced learning rate for stability
        
        # Learning rate scheduler with patience
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True,
            min_lr=1e-5
        )
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 15  # Increased patience
        
        # Training loop with stability measures
        for i in range(self.num_epochs):
            try:
                optimizer.zero_grad()
                
                # Use GPyTorch's built-in stability settings
                with gpytorch.settings.cholesky_jitter(1e-3):
                    with gpytorch.settings.min_variance(1e-4):
                        output = self.model(train_x)
                        loss = mll(output, train_y)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Invalid loss at epoch {i+1}, skipping update...")
                            patience_counter += 1
                            if patience_counter > max_patience:
                                print("Too many invalid losses, stopping training...")
                                break
                            continue
                        
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                        
                        optimizer.step()
                        
                        # Learning rate scheduling
                        scheduler.step(loss)
                        
                        # Early stopping with improved criteria
                        if loss < best_loss * 0.999:  # Allow small improvements
                            best_loss = loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter > max_patience:
                                print("Early stopping triggered...")
                                break
                        
                        if (i+1) % 10 == 0:
                            print(f'Epoch {i+1}/{self.num_epochs} - Loss: {loss.item():.3f}')
                            
            except RuntimeError as e:
                print(f"Error at epoch {i+1}: {str(e)}")
                patience_counter += 1
                if patience_counter > max_patience:
                    print("Too many errors, stopping training...")
                    break
                continue

    def predict(self, X: np.ndarray):
        """Make predictions at test points"""
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
            
        # Scale input features
        X_scaled = self.x_scaler.transform(X)
        test_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            mean = observed_pred.mean.cpu().numpy()
            variance = observed_pred.variance.cpu().numpy()
            
            # Inverse transform predictions
            mean = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
            # Scale back the variance
            variance = variance * (self.y_scaler.scale_ ** 2)
            
        return mean, variance

    def get_params(self):
        """Get the model parameters"""
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        params = {
            'kernel_lengthscale': self.model.rbf_kernel.lengthscale.detach().cpu().numpy(),
            'kernel_outputscale': self.model.covar_module.outputscale.detach().cpu().numpy(),
            'likelihood_noise': self.likelihood.noise.detach().cpu().numpy(),
            'inducing_points': self.model.variational_strategy.inducing_points.detach().cpu().numpy()
        }
        return params