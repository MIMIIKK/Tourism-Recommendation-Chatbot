import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import MinMaxScaler

class SustainabilityFeatureExtractor:
    """Extract and process sustainability-related features"""
    
    def __init__(self, destinations: pd.DataFrame = None, activities: pd.DataFrame = None):
        self.destinations = destinations
        self.activities = activities
        self.scaler = MinMaxScaler()
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load data if not provided"""
        if self.destinations is None:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
        
        if self.activities is None:
            self.activities = pd.read_pickle(f"{processed_dir}/activities.pkl")
    
    def extract_destination_sustainability_features(self) -> np.ndarray:
        """Extract sustainability features from destinations"""
        if self.destinations is None:
            self.load_data()
        
        # Select sustainability-related columns
        sustainability_cols = [
            "carbon_footprint_score", 
            "water_consumption_score",
            "waste_management_score", 
            "biodiversity_impact_score",
            "local_economy_support_score"
        ]
        
        # Extract features
        features = self.destinations[sustainability_cols].values
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        return features_normalized
    
    def extract_activity_sustainability_features(self) -> Dict[int, np.ndarray]:
        """Extract sustainability features from activities for each destination"""
        if self.activities is None:
            self.load_data()
        
        # Select sustainability-related columns
        sustainability_cols = [
            "environmental_impact_score",
            "local_community_benefit_score",
            "resource_consumption_score",
            "eco_friendliness"
        ]
        
        # Group activities by destination
        dest_activities = {}
        
        for dest_id in self.destinations["destination_id"].unique():
            # Filter activities for this destination
            dest_acts = self.activities[self.activities["destination_id"] == dest_id]
            
            if len(dest_acts) > 0:
                # Calculate average sustainability metrics for activities
                avg_metrics = dest_acts[sustainability_cols].mean().values.reshape(1, -1)
                dest_activities[dest_id] = avg_metrics
            else:
                # No activities for this destination
                dest_activities[dest_id] = np.zeros((1, len(sustainability_cols)))
        
        return dest_activities
    
    def combine_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Combine destination and activity sustainability features"""
        if self.destinations is None or self.activities is None:
            self.load_data()
        
        # Extract destination sustainability features
        dest_features = self.extract_destination_sustainability_features()
        
        # Extract activity sustainability features
        activity_features = self.extract_activity_sustainability_features()
        
        # Combine features for each destination
        combined_features = []
        dest_ids = []
        
        for i, dest_id in enumerate(self.destinations["destination_id"]):
            # Get destination features
            dest_feat = dest_features[i].reshape(1, -1)
            
            # Get activity features for this destination
            act_feat = activity_features.get(dest_id, np.zeros((1, 4)))
            
            # Combine features
            combined = np.hstack([dest_feat, act_feat]).flatten()
            
            combined_features.append(combined)
            dest_ids.append(dest_id)
        
        return np.array(combined_features), np.array(dest_ids)
    
    def get_sustainability_feature_names(self) -> List[str]:
        """Get names of sustainability features"""
        dest_features = [
            "carbon_footprint",
            "water_management",
            "waste_management",
            "biodiversity_impact",
            "local_economy"
        ]
        
        activity_features = [
            "avg_environmental_impact",
            "avg_community_benefit",
            "avg_resource_consumption",
            "avg_eco_friendliness"
        ]
        
        return dest_features + activity_features