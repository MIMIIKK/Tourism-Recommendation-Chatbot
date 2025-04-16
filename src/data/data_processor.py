import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import pickle
import os

class DataProcessor:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.processed_dir = "data/processed"
        self.destinations = None
        self.sustainability = None
        self.activities = None
        self.users = None
    
    def load_data(self):
        """Load all raw datasets"""
        self.destinations = pd.read_csv(f"{self.data_dir}/destinations.csv")
        self.sustainability = pd.read_csv(f"{self.data_dir}/sustainability_metrics.csv")
        self.activities = pd.read_csv(f"{self.data_dir}/activities.csv")
        self.users = pd.read_csv(f"{self.data_dir}/users.csv")
        
        print(f"Loaded {len(self.destinations)} destinations")
        print(f"Loaded {len(self.sustainability)} sustainability records")
        print(f"Loaded {len(self.activities)} activities")
        print(f"Loaded {len(self.users)} users")
    
    def preprocess_destinations(self):
        """Process destination data and merge with sustainability metrics"""
        # Merge destination and sustainability data
        self.destinations = pd.merge(
            self.destinations,
            self.sustainability,
            on="destination_id",
            how="left"
        )
        
        # Process categorical features
        for cat_col in ["country", "climate", "landscape_type", "peak_season"]:
            self.destinations[cat_col] = self.destinations[cat_col].astype('category')
        
        # Process list-type columns
        list_cols = ["popular_activities", "accommodation_types", "sustainable_transportation_options"]
        for col in list_cols:
            if col in self.destinations.columns:
                self.destinations[col] = self.destinations[col].str.split(",")
        
        return self.destinations
    
    def preprocess_activities(self):
        """Process activities data"""
        # Create category encoding
        self.activities["category"] = self.activities["category"].astype('category')
        
        # Create season encoding
        season_map = {
            "Year-round": 0,
            "Summer only": 1,
            "Winter only": 2,
            "Spring/Fall": 3,
            "Seasonal": 4
        }
        self.activities["season_code"] = self.activities["seasonal_availability"].map(season_map)
        
        # Calculate overall eco-friendliness score (inverse of impact + benefit)
        self.activities["eco_friendliness"] = (10 - self.activities["environmental_impact_score"]) + \
                                              self.activities["local_community_benefit_score"]
        self.activities["eco_friendliness"] = self.activities["eco_friendliness"] / 2  # Scale to 1-10
        
        return self.activities
    
    def preprocess_users(self):
        """Process user data"""
        # Process list-type columns
        list_cols = ["interests", "travel_history", "preferred_activities"]
        for col in list_cols:
            self.users[col] = self.users[col].str.split(",")
        
        # Convert travel_history from strings to integers
        self.users["travel_history"] = self.users["travel_history"].apply(
            lambda x: [int(i) for i in x] if isinstance(x, list) and x and x[0] else []
        )
        
        # Create user-sustainability profile
        self.users["sustainability_group"] = pd.cut(
            self.users["sustainability_preference"],
            bins=[0, 3, 6, 8, 10],
            labels=["Low", "Medium", "High", "Very High"]
        )
        
        return self.users
    
    def create_user_destination_matrix(self):
        """Create user-destination interaction matrix from travel history"""
        all_user_ids = self.users["user_id"].unique()
        all_dest_ids = self.destinations["destination_id"].unique()
        
        # Initialize interaction matrix
        interactions = np.zeros((len(all_user_ids), len(all_dest_ids)))
        
        # Fill in the matrix based on travel history
        for idx, user in self.users.iterrows():
            user_idx = user["user_id"] - 1  # Adjust for 0-indexing
            
            if isinstance(user["travel_history"], list):
                for dest_id in user["travel_history"]:
                    if dest_id in all_dest_ids:
                        dest_idx = np.where(all_dest_ids == dest_id)[0][0]
                        interactions[user_idx, dest_idx] = 1
        
        return interactions, all_user_ids, all_dest_ids
    
    def extract_destination_features(self):
        """Extract and normalize features for destinations"""
        # Select relevant columns
        feature_cols = [
            "carbon_footprint_score", 
            "water_consumption_score",
            "waste_management_score", 
            "biodiversity_impact_score",
            "local_economy_support_score", 
            "overall_sustainability_score"
        ]
        
        # Extract features
        features = self.destinations[feature_cols].values
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features)
        
        return features_normalized, self.destinations["destination_id"].values
    
    def process_all(self):
        """Process all data and save processed files"""
        os.makedirs(self.processed_dir, exist_ok=True)
        
        print("Loading data...")
        self.load_data()
        
        print("Processing destinations...")
        processed_destinations = self.preprocess_destinations()
        
        print("Processing activities...")
        processed_activities = self.preprocess_activities()
        
        print("Processing users...")
        processed_users = self.preprocess_users()
        
        print("Creating user-destination matrix...")
        interaction_matrix, user_ids, dest_ids = self.create_user_destination_matrix()
        
        print("Extracting destination features...")
        dest_features, dest_ids_features = self.extract_destination_features()
        
        # Save processed data
        with open(f"{self.processed_dir}/interaction_matrix.pkl", "wb") as f:
            pickle.dump({
                "matrix": interaction_matrix,
                "user_ids": user_ids,
                "dest_ids": dest_ids
            }, f)
        
        with open(f"{self.processed_dir}/destination_features.pkl", "wb") as f:
            pickle.dump({
                "features": dest_features,
                "dest_ids": dest_ids_features
            }, f)
        
        # Save processed DataFrames
        processed_destinations.to_pickle(f"{self.processed_dir}/destinations.pkl")
        processed_activities.to_pickle(f"{self.processed_dir}/activities.pkl")
        processed_users.to_pickle(f"{self.processed_dir}/users.pkl")
        
        print("All data processed and saved!")
        
        return {
            "destinations": processed_destinations,
            "activities": processed_activities,
            "users": processed_users,
            "interaction_matrix": interaction_matrix,
            "destination_features": dest_features
        }


if __name__ == "__main__":
    # Process the data
    processor = DataProcessor()
    processed_data = processor.process_all()