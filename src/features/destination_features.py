import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

class DestinationFeatureExtractor:
    """Extract and process features from destination data"""
    
    def __init__(self, destinations: pd.DataFrame = None):
        self.destinations = destinations
        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(stop_words='english')
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load destination data if not provided"""
        if self.destinations is None:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
    
    def extract_numerical_features(self) -> np.ndarray:
        """Extract and normalize numerical features from destinations"""
        if self.destinations is None:
            self.load_data()
        
        # Select relevant numerical columns
        numerical_cols = [
            "carbon_footprint_score", 
            "water_consumption_score",
            "waste_management_score", 
            "biodiversity_impact_score",
            "local_economy_support_score", 
            "overall_sustainability_score"
        ]
        
        # Extract features
        features = self.destinations[numerical_cols].values
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        return features_normalized
    
    def extract_categorical_features(self) -> Dict[str, np.ndarray]:
        """Extract and one-hot encode categorical features"""
        if self.destinations is None:
            self.load_data()
        
        # Select relevant categorical columns
        cat_cols = ["country", "climate", "landscape_type", "peak_season"]
        
        # One-hot encode each categorical feature
        encoded_features = {}
        
        for col in cat_cols:
            # Get dummies (one-hot encoding)
            dummies = pd.get_dummies(self.destinations[col], prefix=col)
            encoded_features[col] = dummies.values
        
        return encoded_features
    
    def extract_text_features(self, text_col: str = "popular_activities") -> np.ndarray:
        """Extract features from text data using TF-IDF"""
        if self.destinations is None:
            self.load_data()
        
        # Handle list columns by joining elements
        if isinstance(self.destinations[text_col].iloc[0], list):
            text_data = self.destinations[text_col].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
        else:
            text_data = self.destinations[text_col]
        
        # Apply TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(text_data)
        
        return tfidf_matrix
    
    def combine_features(self, include_text: bool = True) -> np.ndarray:
        """Combine different feature types into a single feature matrix"""
        # Extract different feature types
        numerical = self.extract_numerical_features()
        categorical = self.extract_categorical_features()
        
        # Combine categorical features
        cat_combined = np.hstack([categorical[col] for col in categorical])
        
        # Combine numerical and categorical
        features = np.hstack([numerical, cat_combined])
        
        # Add text features if requested
        if include_text:
            text_features = self.extract_text_features().toarray()
            features = np.hstack([features, text_features])
        
        return features
    
    def extract_features(self, include_text: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Extract all features and return with destination IDs"""
        if self.destinations is None:
            self.load_data()
        
        features = self.combine_features(include_text)
        dest_ids = self.destinations["destination_id"].values
        
        return features, dest_ids