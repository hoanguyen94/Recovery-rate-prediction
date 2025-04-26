import numpy as np
import pandas as pd
import shap
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Union

class TabPFNShapExplainer:
    """
    A class to compute SHAP values for TabPFN regression models.
    """
    def __init__(self, model: TabPFNRegressor, background_data: pd.DataFrame, scaler: StandardScaler):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained TabPFNRegressor model
            background_data: Background dataset for SHAP computation
            scaler: Fitted StandardScaler used to scale the features
        """
        self.model = model
        self.background_data = background_data
        self.scaler = scaler
        
        # Scale background data
        self.background_scaled = self.scaler.transform(background_data)
        
        # Create a wrapper function for the model's predict method
        def model_predict(x):
            return self.model.predict(x)
        
        # Initialize the explainer
        self.explainer = shap.KernelExplainer(
            model_predict,
            self.background_scaled,
            link="identity"
        )
    
    def explain(self, data: pd.DataFrame, nsamples: int = 100) -> Tuple[np.ndarray, List[str]]:
        """
        Compute SHAP values for the given data.
        
        Args:
            data: Data to explain
            nsamples: Number of samples to use for SHAP computation
            
        Returns:
            Tuple containing:
            - SHAP values array
            - Feature names list
        """
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(
            data_scaled,
            nsamples=nsamples
        )
        
        return shap_values, data.columns.tolist()
    
    def get_feature_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Compute feature importance based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        # Compute mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of feature importance
        importance_dict = dict(zip(feature_names, mean_abs_shap))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
    
    def plot_summary(self, shap_values: np.ndarray, feature_names: List[str], max_display: int = 20):
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            max_display: Maximum number of features to display
        """
        shap.summary_plot(
            shap_values,
            self.background_data,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
    
    def plot_dependence(self, shap_values: np.ndarray, feature_names: List[str], feature_idx: int):
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            feature_idx: Index of the feature to plot
        """
        shap.dependence_plot(
            feature_idx,
            shap_values,
            self.background_data,
            feature_names=feature_names,
            show=False
        ) 