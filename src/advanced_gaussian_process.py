import torch
import gpytorch
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import os

class FeatureProcessor:
    def __init__(self, numerical_features: List[str], poly_degree: int = 2, n_best_features: int = 50):
        """
        Initialize FeatureProcessor.
        
        Args:
            numerical_features: List of numerical feature names
            poly_degree: Degree of polynomial features
            n_best_features: Number of best features to select
        """
        self.numerical_features = numerical_features
        self.poly_degree = poly_degree
        self.n_best_features = n_best_features
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.selector = SelectKBest(mutual_info_regression, k=n_best_features)
        
        # Initialize ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features)
            ],
            remainder='passthrough'  # Leave categorical features untouched
        )
        
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform the data with proper scaling for numerical and categorical features.
        
        Args:
            X: Input DataFrame
            y: Target values (required for feature selection)
        """
        # 1. Initial preprocessing (scaling)
        X_scaled = self.preprocessor.fit_transform(X)
        
        # Convert to DataFrame to maintain feature names
        scaled_features = pd.DataFrame(
            X_scaled,
            columns=self.numerical_features + [col for col in X.columns 
                                             if col not in self.numerical_features]
        )
        
        # 2. Generate polynomial features for numerical features only
        num_features_scaled = scaled_features[self.numerical_features].values
        X_poly = self.poly.fit_transform(num_features_scaled)
        
        # Create feature names for polynomial features
        poly_feature_names = self.poly.get_feature_names_out(self.numerical_features)
        
        # 3. Create interaction terms for numerical features
        interactions = []
        interaction_names = []
        for i, j in combinations(range(len(self.numerical_features)), 2):
            interaction = num_features_scaled[:, i] * num_features_scaled[:, j]
            interactions.append(interaction)
            interaction_names.append(f"{self.numerical_features[i]}_{self.numerical_features[j]}_interaction")
        
        if interactions:
            interactions = np.column_stack(interactions)
        
        # 4. Non-linear transformations for numerical features
        nonlinear = np.column_stack([
            np.log1p(np.abs(num_features_scaled)),
            np.sin(num_features_scaled),
            np.cos(num_features_scaled),
            num_features_scaled**2
        ])
        
        nonlinear_names = []
        for feat in self.numerical_features:
            nonlinear_names.extend([
                f"{feat}_log",
                f"{feat}_sin",
                f"{feat}_cos",
                f"{feat}_squared"
            ])
        
        # 5. Combine all features
        categorical_features = scaled_features.drop(columns=self.numerical_features)
        
        X_transformed = np.column_stack([
            X_poly,  # Polynomial features
            interactions,  # Interaction terms
            nonlinear,  # Non-linear transformations
            categorical_features  # Original categorical features
        ])
        
        # Combine all feature names
        all_feature_names = list(poly_feature_names) + interaction_names + \
                          nonlinear_names + list(categorical_features.columns)
        
        # 6. Feature selection - no need for y check since this is fit_transform
        X_selected = self.selector.fit_transform(X_transformed, y)
        # Store selected feature indices and names
        self.selected_features_mask = self.selector.get_support()
        self.selected_feature_names = [name for name, selected in 
                                     zip(all_feature_names, self.selected_features_mask) 
                                     if selected]
        
        return X_selected
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data during prediction.
        
        Args:
            X: Input DataFrame
        """
        # 1. Initial preprocessing (scaling)
        X_scaled = self.preprocessor.transform(X)
        
        # Convert to DataFrame to maintain feature names
        scaled_features = pd.DataFrame(
            X_scaled,
            columns=self.numerical_features + [col for col in X.columns 
                                             if col not in self.numerical_features]
        )
        
        # 2. Generate polynomial features for numerical features only
        num_features_scaled = scaled_features[self.numerical_features].values
        X_poly = self.poly.transform(num_features_scaled)
        
        # 3. Create interaction terms for numerical features
        interactions = []
        for i, j in combinations(range(len(self.numerical_features)), 2):
            interaction = num_features_scaled[:, i] * num_features_scaled[:, j]
            interactions.append(interaction)
        
        if interactions:
            interactions = np.column_stack(interactions)
        
        # 4. Non-linear transformations for numerical features
        nonlinear = np.column_stack([
            np.log1p(np.abs(num_features_scaled)),
            np.sin(num_features_scaled),
            np.cos(num_features_scaled),
            num_features_scaled**2
        ])
        
        # 5. Combine all features
        categorical_features = scaled_features.drop(columns=self.numerical_features)
        
        X_transformed = np.column_stack([
            X_poly,
            interactions,
            nonlinear,
            categorical_features
        ])
        
        # 6. Feature selection
        X_selected = self.selector.transform(X_transformed)
        
        return X_selected
    
    def get_feature_names(self) -> List[str]:
        """Get names of selected features after feature selection."""
        if not hasattr(self, 'selected_feature_names'):
            raise RuntimeError("Feature names not available. Call fit_transform first.")
        return self.selected_feature_names

class DeepFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [100, 50, 20]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AdvancedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods.Likelihood, using_deep_feature: bool = True):
        super().__init__(train_x, train_y, likelihood)
        
        input_dim = train_x.size(1)
        hidden_dims = [100, 50, 20]
        output_dim = hidden_dims[-1]
        self.feature_extractor = DeepFeatureExtractor(input_dim, hidden_dims)
        self.using_deep_feature = using_deep_feature
        
        # Mean module - combination of constant and linear mean
        
        if not self.using_deep_feature:
           output_dim = train_x.size(1)
        # self.mean_module = gpytorch.means.LinearMean(output_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Composite kernel
        # RBF kernel for smooth functions
        self.rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=output_dim)

        # Matern kernel for less smooth functions
        self.matern_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=output_dim)

        # Periodic kernel for periodic functions
        # self.periodic_kernel = gpytorch.kernels.PeriodicKernel(ard_num_dims=output_dim)

        # Spectral mixture kernel for complex functions
        # self.spectral_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=output_dim)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.rbf_kernel + self.matern_kernel
        )
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     self.rbf_kernel
        # )
        self.covar_module.outputscale = 1.0
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        if self.using_deep_feature:
            features = self.feature_extractor(x)
        else:
            features = x
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class AdvancedGaussianProcess:
    def __init__(self, 
                 numerical_features: List[str],
                 num_epochs: int = 200,
                 learning_rate: float = 0.01,
                 poly_degree: int = 2,
                 n_best_features: int = 50,
                 using_feature_processor: bool = False,
                 using_deep_feature: bool = True):
        """
        Initialize Advanced Gaussian Process model.
        
        Args:
            numerical_features: List of numerical feature names
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            poly_degree: Degree of polynomial features
            n_best_features: Number of best features to select
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.using_feature_processor = using_feature_processor
        if using_feature_processor:
            self.feature_processor = FeatureProcessor(
                numerical_features=numerical_features,
                poly_degree=poly_degree,
                n_best_features=n_best_features
            )
        else:
            self.feature_processor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features)   # StandardScaler()
            ],
            remainder='passthrough'  # Leave categorical features untouched
        )
        self.using_deep_feature = using_deep_feature
        self.model = None
        self.likelihood = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_scaler = StandardScaler()  # Add Y scaler
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the Advanced Gaussian Process model to the training data.
        
        Args:
            X: Input features as DataFrame
            y: Target values
        """
        # Process features
        X_processed = self.feature_processor.fit_transform(X, y)
        
        # Standardize Y
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Convert to tensors
        X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        # Initialize model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
          # Set initial noise
        self.likelihood.noise = torch.tensor(1e-4)
        self.model = AdvancedGPModel(X_tensor, y_tensor, self.likelihood, self.using_deep_feature).to(self.device)
        
        # Training mode
        self.model.train()
        self.likelihood.train()
        
        # Optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()}
        ], lr=self.learning_rate)
        
        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch+1}/{self.num_epochs} - Loss: {loss.item():.3f}')
    
    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the trained model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Process features
        X_processed = self.feature_processor.transform(X)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_tensor))
            mean = predictions.mean.cpu().numpy()
            
            # Inverse transform predictions back to original scale
            mean = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
            
            if return_std:
                std = predictions.stddev.cpu().numpy()
                # Scale back the standard deviation
                std = std * self.y_scaler.scale_  # Adjust std for original scale
                return mean, std
            
            return mean, None
    
    def get_params(self) -> dict:
        """Get the learned hyperparameters of the model."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        params = {
            'kernel_rbf_lengthscale': self.model.rbf_kernel.lengthscale.detach().cpu().numpy(),
            'kernel_matern_lengthscale': self.model.matern_kernel.lengthscale.detach().cpu().numpy(),
            'kernel_periodic_lengthscale': self.model.periodic_kernel.lengthscale.detach().cpu().numpy(),
            'kernel_periodic_period': self.model.periodic_kernel.period_length.detach().cpu().numpy(),
            'kernel_scale': self.model.covar_module.outputscale.detach().cpu().numpy(),
            'likelihood_noise': self.likelihood.noise.detach().cpu().numpy()
        }

         # Add feature extractor parameters
        if self.using_feature_processor:
            feature_extractor_params = {}
            for name, layer in self.model.feature_extractor.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    feature_extractor_params[f'fe_{name}_weight'] = layer.weight.detach().cpu().numpy()
                    feature_extractor_params[f'fe_{name}_bias'] = layer.bias.detach().cpu().numpy()
                elif isinstance(layer, torch.nn.BatchNorm1d):
                    feature_extractor_params[f'fe_{name}_weight'] = layer.weight.detach().cpu().numpy()
                    feature_extractor_params[f'fe_{name}_bias'] = layer.bias.detach().cpu().numpy()
                    feature_extractor_params[f'fe_{name}_running_mean'] = layer.running_mean.detach().cpu().numpy()
                    feature_extractor_params[f'fe_{name}_running_var'] = layer.running_var.detach().cpu().numpy()
        
        params.update(feature_extractor_params)
        return params

    def get_feature_importances(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance from multiple sources in the advanced GP model.
        
        Args:
            feature_names: Optional list of feature names. If None, will use feature_0, feature_1, etc.
            
        Returns:
            DataFrame containing feature importance scores from different sources
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # 1. Get importance from RBF kernel lengthscales
        rbf_lengthscales = self.model.rbf_kernel.lengthscale.detach().cpu().numpy()
        if len(rbf_lengthscales.shape) > 1:
            rbf_lengthscales = rbf_lengthscales.squeeze()
        rbf_importance = 1.0 / rbf_lengthscales
        rbf_importance = rbf_importance / rbf_importance.sum()
        
        # 2. Get importance from Matern kernel lengthscales
        matern_lengthscales = self.model.matern_kernel.lengthscale.detach().cpu().numpy()
        if len(matern_lengthscales.shape) > 1:
            matern_lengthscales = matern_lengthscales.squeeze()
        matern_importance = 1.0 / matern_lengthscales
        matern_importance = matern_importance / matern_importance.sum()
        
        # 3. Get importance from periodic kernel lengthscales
        periodic_lengthscales = self.model.periodic_kernel.lengthscale.detach().cpu().numpy()
        if len(periodic_lengthscales.shape) > 1:
            periodic_lengthscales = periodic_lengthscales.squeeze()
        periodic_importance = 1.0 / periodic_lengthscales
        periodic_importance = periodic_importance / periodic_importance.sum()
        
        # 4. Get importance from feature extractor weights
        feature_extractor_weights = []
        for layer in self.model.feature_extractor.network:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().cpu().numpy()
                importance = np.abs(weights).mean(axis=0)
                feature_extractor_weights.append(importance)
        
        # Use first layer weights as feature importance
        nn_importance = feature_extractor_weights[0]
        nn_importance = nn_importance / nn_importance.sum()
        
        # 5. Get feature selection importance from the processor
        if hasattr(self.feature_processor.selector, 'scores_'):
            selection_importance = self.feature_processor.selector.scores_
            selection_importance = selection_importance / selection_importance.sum()
        else:
            selection_importance = np.ones_like(rbf_importance) / len(rbf_importance)
        
        # Combine all importance scores (weighted average)
        weights = {
            'rbf': 0.3,
            'matern': 0.2,
            'periodic': 0.1,
            'neural_network': 0.2,
            'feature_selection': 0.2
        }
        
        combined_importance = (
            weights['rbf'] * rbf_importance +
            weights['matern'] * matern_importance +
            weights['periodic'] * periodic_importance +
            weights['neural_network'] * nn_importance +
            weights['feature_selection'] * selection_importance
        )
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(combined_importance))]
        
        # Create DataFrame with all importance scores
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Combined_Importance': combined_importance,
            'RBF_Importance': rbf_importance,
            'Matern_Importance': matern_importance,
            'Periodic_Importance': periodic_importance,
            'NN_Importance': nn_importance,
            'Selection_Importance': selection_importance
        })
        
        return importance_df.sort_values('Combined_Importance', ascending=False)

    def plot_feature_importances(self, 
                               feature_names: Optional[list] = None, 
                               top_k: Optional[int] = None,
                               save_path: Optional[str] = '../output/feature_importance',
                               dpi: int = 300):
        """
        Plot feature importance from all sources and save the figures.
        
        Args:
            feature_names: Optional list of feature names
            top_k: Optional number of top features to show
            save_path: Base path to save the figures (without extension)
            dpi: Resolution of saved figures
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get feature importance
        importance_df = self.get_feature_importances(feature_names)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        # Create two separate figures for better quality
        # Figure 1: Combined Importance
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        importance_df.plot(
            x='Feature',
            y='Combined_Importance',
            kind='bar',
            ax=ax1,
            title='Combined Feature Importance'
        )
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        # Save combined importance plot
        combined_path = f"{save_path}_combined.png"
        plt.savefig(combined_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved combined importance plot to: {combined_path}")
        plt.close()
        
        # Figure 2: Individual Sources
        plt.figure(figsize=(12, 6))
        ax2 = plt.gca()
        importance_sources = [
            'RBF_Importance', 'Matern_Importance', 'Periodic_Importance',
            'NN_Importance', 'Selection_Importance'
        ]
        
        importance_df[['Feature'] + importance_sources].set_index('Feature').plot(
            kind='bar',
            ax=ax2,
            title='Feature Importance by Source'
        )
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        # Save individual sources plot
        sources_path = f"{save_path}_by_source.png"
        plt.savefig(sources_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved source-wise importance plot to: {sources_path}")
        plt.close()
        
        # Save importance data to CSV
        csv_path = f"{save_path}_data.csv"
        importance_df.to_csv(csv_path, index=False)
        print(f"Saved importance data to: {csv_path}")
        
        # Create a heatmap of feature importance
        plt.figure(figsize=(12, 8))
        importance_matrix = importance_df[importance_sources].values
        im = plt.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, label='Importance Score')
        
        # Set labels
        plt.yticks(range(len(importance_sources)), importance_sources)
        plt.xticks(range(len(importance_df)), importance_df['Feature'], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance Source')
        plt.title('Feature Importance Heatmap')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = f"{save_path}_heatmap.png"
        plt.savefig(heatmap_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved importance heatmap to: {heatmap_path}")
        plt.close()
        
        # Return the DataFrame for further analysis if needed
        return importance_df

    def get_top_features(self, feature_names: Optional[list] = None, top_k: int = 10) -> list:
        """
        Get the top k most important features.
        
        Args:
            feature_names: Optional list of feature names
            top_k: Number of top features to return
            
        Returns:
            List of top feature names
        """
        importance_df = self.get_feature_importances(feature_names)
        return importance_df.head(top_k)['Feature'].tolist()

class GPEnsemble:
    def __init__(self, 
                 numerical_features: List[str],
                 n_models: int = 5,
                 num_epochs: int = 200,
                 learning_rate: float = 0.01,
                 poly_degree: int = 2,
                 n_best_features: int = 50,
                 bagging_fraction: float = 0.8,
                 feature_fraction: float = 0.8,
                 using_deep_feature: bool = True,
                 using_feature_processor: bool = False):
        """
        Initialize GP Ensemble.
        
        Args:
            numerical_features: List of numerical feature names
            n_models: Number of GP models in ensemble
            num_epochs: Number of training epochs per model
            learning_rate: Learning rate for optimization
            poly_degree: Degree of polynomial features
            n_best_features: Number of features to select
            bagging_fraction: Fraction of data to use for each model
            feature_fraction: Fraction of features to use for each model
        """
        self.numerical_features = numerical_features
        self.n_models = n_models
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.poly_degree = poly_degree
        self.n_best_features = n_best_features
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.using_deep_feature = using_deep_feature
        self.using_feature_processor = using_feature_processor
        self.models = []
        self.feature_masks = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_feature_mask(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Create random feature mask for both numerical and categorical features.
        
        Args:
            X: Input DataFrame with all features
            
        Returns:
            selected_num_features: List of selected numerical features
            selected_cat_features: List of selected categorical features
        """
        all_features = X.columns.tolist()
        categorical_features = [f for f in all_features if f not in self.numerical_features]
        
        # Number of features to select
        n_num_features = int(len(self.numerical_features) * self.feature_fraction)
        n_cat_features = int(len(categorical_features) * self.feature_fraction)
        
        # Randomly select features
        selected_num_features = np.random.choice(
            self.numerical_features, 
            size=n_num_features, 
            replace=False
        ).tolist()
        
        selected_cat_features = np.random.choice(
            categorical_features, 
            size=n_cat_features, 
            replace=False
        ).tolist()
        
        return selected_num_features, selected_cat_features
    
    def _get_bagging_indices(self, n_samples: int) -> np.ndarray:
        """Get random indices for bagging."""
        n_selected = int(n_samples * self.bagging_fraction)
        return np.random.choice(n_samples, n_selected, replace=True)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit ensemble of GP models.
        
        Args:
            X: Input DataFrame with both numerical and categorical features
            y: Target values
        """
        n_samples = len(X)
        
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # Create feature masks for both numerical and categorical features
            selected_num_features, selected_cat_features = self._create_feature_mask(X)
            selected_features = selected_num_features + selected_cat_features
            
            # Store the feature mask
            self.feature_masks.append(selected_features)
            
            # Get bagging indices
            bag_indices = self._get_bagging_indices(n_samples)
            
            # Create and train model on subset of data and features
            model = AdvancedGaussianProcess(
                numerical_features=selected_num_features,  # Just pass the list directly
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                poly_degree=self.poly_degree,
                n_best_features=self.n_best_features,
                using_deep_feature=self.using_deep_feature,
                using_feature_processor=self.using_feature_processor
            )
            
            # Select subset of data and features
            X_subset = X.iloc[bag_indices][selected_features]
            y_subset = y[bag_indices]
            
            model.fit(X_subset, y_subset)
            self.models.append(model)
    
    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Input features (n_samples, n_features)
            return_std: Whether to return standard deviation
            
        Returns:
            mean: Mean predictions
            std: Standard deviation of predictions (if return_std=True)
        """
        predictions = []
        uncertainties = []
        
        # Get predictions from each model
        for model, selected_features in zip(self.models, self.feature_masks):
            X_subset = X[selected_features]
            if return_std:
                mean, std = model.predict(X_subset, return_std=True)
                predictions.append(mean)
                uncertainties.append(std)
            else:
                mean, _ = model.predict(X_subset, return_std=False)
                predictions.append(mean)
        
        # Stack predictions
        predictions = np.stack(predictions)
        
        # Calculate ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        if return_std:
            # Calculate total uncertainty:
            # 1. Aleatoric uncertainty (mean of individual model uncertainties)
            aleatoric_uncertainty = np.mean(np.stack(uncertainties), axis=0)
            
            # 2. Epistemic uncertainty (standard deviation of predictions)
            epistemic_uncertainty = np.std(predictions, axis=0)
            
            # 3. Total uncertainty (combine both sources)
            total_uncertainty = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
            
            return ensemble_mean, total_uncertainty
        
        return ensemble_mean, None
    
    def get_feature_importances(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Calculate feature importance based on usage frequency in the ensemble.
        
        Args:
            feature_names: List of feature names (optional)
            
        Returns:
            DataFrame with feature importances
        """
        feature_masks = np.stack(self.feature_masks)
        importance_scores = np.mean(feature_masks, axis=0)
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance_scores))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        
        return importance_df.sort_values('Importance', ascending=False)

# Example usage class that combines both individual GP and ensemble
class AdvancedGPWithEnsemble:
    def __init__(self, 
                 numerical_features: List[str],
                 use_ensemble: bool = True,
                 n_models: int = 5,
                 num_epochs: int = 200,
                 learning_rate: float = 0.01,
                 poly_degree: int = 2,
                 n_best_features: int = 50,
                 bagging_fraction: float = 0.8,
                 feature_fraction: float = 0.8,
                 using_deep_feature: bool = True,
                 using_feature_processor: bool = False):
        """
        Initialize either single GP or GP ensemble.
        
        Args:
            numerical_features: List of numerical feature names
            use_ensemble: Whether to use ensemble or single model
            Other parameters same as GPEnsemble
        """
        self.use_ensemble = use_ensemble
        if use_ensemble:
            self.model = GPEnsemble(
                numerical_features=numerical_features,
                n_models=n_models,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                poly_degree=poly_degree,
                n_best_features=n_best_features,
                bagging_fraction=bagging_fraction,
                feature_fraction=feature_fraction,
                using_deep_feature=using_deep_feature,
                using_feature_processor=using_feature_processor
            )
        else:
            self.model = AdvancedGaussianProcess(
                numerical_features=numerical_features,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                poly_degree=poly_degree,
                n_best_features=n_best_features,
                using_feature_processor=using_feature_processor,
                using_deep_feature=using_deep_feature
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions."""
        return self.model.predict(X, return_std=return_std)
    
    def get_feature_importances(self, feature_names: Optional[list] = None) -> Optional[pd.DataFrame]:
        """Get feature importances (only for ensemble)."""
        if self.use_ensemble:
            return self.model.get_feature_importances(feature_names)
        else:
            print("Feature importances only available for ensemble model")
            return None