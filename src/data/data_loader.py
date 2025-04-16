import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Any

class DataLoader:
    """Utility class for loading processed data"""
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
    
    def load_all(self) -> Dict[str, Any]:
        """Load all processed data"""
        # Load DataFrames
        destinations = pd.read_pickle(f"{self.processed_dir}/destinations.pkl")
        activities = pd.read_pickle(f"{self.processed_dir}/activities.pkl")
        users = pd.read_pickle(f"{self.processed_dir}/users.pkl")
        
        # Load interaction matrix
        with open(f"{self.processed_dir}/interaction_matrix.pkl", "rb") as f:
            interaction_data = pickle.load(f)
            interaction_matrix = interaction_data["matrix"]
            user_ids = interaction_data["user_ids"]
            dest_ids = interaction_data["dest_ids"]
        
        # Load destination features
        with open(f"{self.processed_dir}/destination_features.pkl", "rb") as f:
            feature_data = pickle.load(f)
            destination_features = feature_data["features"]
            dest_feature_ids = feature_data["dest_ids"]
        
        return {
            "destinations": destinations,
            "activities": activities,
            "users": users,
            "interaction_matrix": interaction_matrix,
            "user_ids": user_ids,
            "dest_ids": dest_ids,
            "destination_features": destination_features,
            "dest_feature_ids": dest_feature_ids
        }
    
    def load_interaction_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load user-destination interaction matrix"""
        with open(f"{self.processed_dir}/interaction_matrix.pkl", "rb") as f:
            data = pickle.load(f)
            return data["matrix"], data["user_ids"], data["dest_ids"]
    
    def load_destination_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load destination features"""
        with open(f"{self.processed_dir}/destination_features.pkl", "rb") as f:
            data = pickle.load(f)
            return data["features"], data["dest_ids"]