import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Tuple, Any
import pickle
import os

class ShapExplainer:
    """
    Class for generating SHAP (SHapley Additive exPlanations) explanations
    for recommendation models
    """
    
    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.background_data = None
    
    def create_explainer(self, background_data: np.ndarray, model_type: str = "tree"):
        """
        Create a SHAP explainer for the model
        
        Parameters:
        - background_data: Background data for the explainer
        - model_type: Type of model ("tree", "linear", "kernel", "deep")
        """
        self.background_data = background_data
        
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model, background_data)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, background_data)
        elif model_type == "kernel":
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        elif model_type == "deep":
            self.explainer = shap.DeepExplainer(self.model, background_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def calculate_shap_values(self, data: np.ndarray):
        """Calculate SHAP values for data"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        self.shap_values = self.explainer.shap_values(data)
        return self.shap_values
    
    def explain_recommendation(self, user_idx: int, dest_idx: int, 
                              feature_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate explanation for a specific recommendation
        
        Parameters:
        - user_idx: User index
        - dest_idx: Destination index
        - feature_data: Feature data for the user-destination pair
        
        Returns:
        - Dictionary with explanation data
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        # Calculate SHAP values for this instance
        shap_values = self.explainer.shap_values(feature_data)
        
        # If feature names not provided, generate generic names
        if self.feature_names is None:
            self.feature_names = [f"Feature {i}" for i in range(feature_data.shape[1])]
        
        # Create a dictionary mapping feature names to SHAP values
        feature_contributions = {}
        
        if isinstance(shap_values, list):
            # For multi-output models, use the output corresponding to the prediction
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            values = shap_values
        
        for i, name in enumerate(self.feature_names):
            feature_contributions[name] = values[0, i]
        
        # Sort features by importance (absolute SHAP value)
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top contributing features
        top_features = sorted_contributions[:5]
        
        # Calculate base value (average prediction)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0.5  # Fallback value
        
        return {
            "user_idx": user_idx,
            "dest_idx": dest_idx,
            "feature_contributions": feature_contributions,
            "sorted_contributions": sorted_contributions,
            "top_features": top_features,
            "base_value": base_value
        }
    
    def generate_feature_importance_plot(self, save_path: str = None):
        """Generate summary plot of feature importance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, 
            self.background_data, 
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            return plt
    
    def generate_dependence_plot(self, feature_idx: int, interaction_idx: int = None, 
                                save_path: str = None):
        """Generate dependence plot for a feature"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        plt.figure(figsize=(10, 6))
        
        feature_name = self.feature_names[feature_idx] if self.feature_names else None
        interaction_name = self.feature_names[interaction_idx] if interaction_idx is not None and self.feature_names else None
        
        shap.dependence_plot(
            feature_idx, 
            self.shap_values, 
            self.background_data,
            interaction_index=interaction_idx,
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            return plt
    
    def save_explainer(self, filepath: str = "models/shap_explainer.pkl"):
        """Save SHAP explainer to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create a dictionary with the explainer and metadata
        explainer_data = {
            "explainer": self.explainer,
            "feature_names": self.feature_names,
            "background_data_shape": self.background_data.shape if self.background_data is not None else None
        }
        
        # Save to pickle file
        with open(filepath, "wb") as f:
            pickle.dump(explainer_data, f)
    
    def load_explainer(self, filepath: str = "models/shap_explainer.pkl"):
        """Load SHAP explainer from file"""
        with open(filepath, "rb") as f:
            explainer_data = pickle.load(f)
        
        self.explainer = explainer_data["explainer"]
        self.feature_names = explainer_data["feature_names"]
        
        return self