import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import copy

class CounterfactualExplainer:
    """
    Generate counterfactual explanations for recommendations.
    
    A counterfactual explanation shows how a recommendation would change
    if certain features or inputs were different.
    """
    
    def __init__(self, recommender=None, destinations: pd.DataFrame = None):
        self.recommender = recommender
        self.destinations = destinations
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load data if not provided"""
        if self.destinations is None:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
    
    def generate_sustainability_counterfactual(self, user_id: int, dest_id: int, 
                                             sustainability_weight: float = 0.3,
                                             target_weight: float = 0.7) -> Dict[str, Any]:
        """
        Generate a counterfactual explanation showing how recommendations
        would change with different sustainability weight
        
        Parameters:
        - user_id: User ID
        - dest_id: Destination ID of current recommendation
        - sustainability_weight: Current sustainability weight
        - target_weight: Target sustainability weight for counterfactual
        
        Returns:
        - Dictionary with counterfactual explanation
        """
        if self.recommender is None:
            raise ValueError("Recommender not provided or initialized")
        
        if self.destinations is None:
            self.load_data()
        
        # Get current rank of the destination
        current_recommendations = self.recommender.recommend(
            user_id, n=20, exclude_visited=True
        )
        
        current_rank = None
        for i, rec in enumerate(current_recommendations):
            if rec["destination_id"] == dest_id:
                current_rank = i + 1
                break
        
        if current_rank is None:
            return {"error": f"Destination ID {dest_id} not found in current recommendations"}
        
        # Store original weight
        original_weight = self.recommender.sustainability_weight
        
        # Set counterfactual weight
        self.recommender.sustainability_weight = target_weight
        
        # Get counterfactual recommendations
        counterfactual_recommendations = self.recommender.recommend(
            user_id, n=20, exclude_visited=True
        )
        
        # Reset original weight
        self.recommender.sustainability_weight = original_weight
        
        # Find new rank of the destination
        new_rank = None
        for i, rec in enumerate(counterfactual_recommendations):
            if rec["destination_id"] == dest_id:
                new_rank = i + 1
                break
        
        if new_rank is None:
            new_rank = "Not in top 20"
        
        # Get destination details
        dest_info = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
        sustainability_score = dest_info["overall_sustainability_score"]
        
        # Find destinations that moved above/below this one
        moved_up = []
        moved_down = []
        
        if new_rank != "Not in top 20" and current_rank != new_rank:
            if new_rank < current_rank:  # Improved rank
                for i in range(new_rank - 1, current_rank - 1):
                    if i < len(current_recommendations):
                        moved_down.append({
                            "name": counterfactual_recommendations[i]["name"],
                            "sustainability_score": counterfactual_recommendations[i]["sustainability_score"]
                        })
            else:  # Worse rank
                for i in range(current_rank, new_rank):
                    if i < len(counterfactual_recommendations):
                        moved_up.append({
                            "name": counterfactual_recommendations[i-1]["name"],
                            "sustainability_score": counterfactual_recommendations[i-1]["sustainability_score"]
                        })
        
        # Create explanation
        explanation = {
            "destination_id": dest_id,
            "destination_name": dest_info["name"],
            "sustainability_score": sustainability_score,
            "current_weight": sustainability_weight,
            "counterfactual_weight": target_weight,
            "current_rank": current_rank,
            "counterfactual_rank": new_rank,
            "rank_change": current_rank - new_rank if new_rank != "Not in top 20" else "Dropped out",
            "moved_up": moved_up,
            "moved_down": moved_down
        }
        
        return explanation
    
    def generate_feature_counterfactual(self, user_id: int, dest_id: int, 
                                      feature: str, target_value: Any) -> Dict[str, Any]:
        """
        Generate a counterfactual explanation showing how recommendations
        would change if a destination feature were different
        
        Parameters:
        - user_id: User ID
        - dest_id: Destination ID
        - feature: Feature to modify
        - target_value: Target value for the feature
        
        Returns:
        - Dictionary with counterfactual explanation
        """
        if self.recommender is None:
            raise ValueError("Recommender not provided or initialized")
        
        if self.destinations is None:
            self.load_data()
        
        # Check if feature exists
        if feature not in self.destinations.columns:
            return {"error": f"Feature '{feature}' not found in destination data"}
        
        # Get current rank of the destination
        current_recommendations = self.recommender.recommend(
            user_id, n=20, exclude_visited=True
        )
        
        current_rank = None
        for i, rec in enumerate(current_recommendations):
            if rec["destination_id"] == dest_id:
                current_rank = i + 1
                break
        
        if current_rank is None:
            return {"error": f"Destination ID {dest_id} not found in current recommendations"}
        
        # Get destination details
        dest_info = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
        current_value = dest_info[feature]
        
        # Create a modified copy of the destinations dataframe
        modified_destinations = self.destinations.copy()
        modified_destinations.loc[modified_destinations["destination_id"] == dest_id, feature] = target_value
        
        # Store original destinations
        original_destinations = self.recommender.destinations
        
        # Set modified destinations
        self.recommender.destinations = modified_destinations
        
        # Get counterfactual recommendations
        counterfactual_recommendations = self.recommender.recommend(
            user_id, n=20, exclude_visited=True
        )
        
        # Reset original destinations
        self.recommender.destinations = original_destinations
        
        # Find new rank of the destination
        new_rank = None
        for i, rec in enumerate(counterfactual_recommendations):
            if rec["destination_id"] == dest_id:
                new_rank = i + 1
                break
        
        if new_rank is None:
            new_rank = "Not in top 20"
        
        # Create explanation
        explanation = {
            "destination_id": dest_id,
            "destination_name": dest_info["name"],
            "feature": feature,
            "current_value": current_value,
            "counterfactual_value": target_value,
            "current_rank": current_rank,
            "counterfactual_rank": new_rank,
            "rank_change": current_rank - new_rank if new_rank != "Not in top 20" else "Dropped out"
        }
        
        return explanation
    
    def generate_user_counterfactual(self, user_id: int, dest_id: int, 
                                   user_feature: str, target_value: Any) -> Dict[str, Any]:
        """
        Generate a counterfactual explanation showing how recommendations
        would change if a user feature were different
        
        Parameters:
        - user_id: User ID
        - dest_id: Destination ID
        - user_feature: User feature to modify
        - target_value: Target value for the feature
        
        Returns:
        - Dictionary with counterfactual explanation
        """
        if self.recommender is None:
            raise ValueError("Recommender not provided or initialized")
        
        # Get current rank of the destination
        current_recommendations = self.recommender.recommend(
            user_id, n=20, exclude_visited=True
        )
        
        current_rank = None
        for i, rec in enumerate(current_recommendations):
            if rec["destination_id"] == dest_id:
                current_rank = i + 1
                break
        
        if current_rank is None:
            return {"error": f"Destination ID {dest_id} not found in current recommendations"}
        
        # Check if user feature exists
        if user_feature not in self.recommender.users.columns:
            return {"error": f"User feature '{user_feature}' not found in user data"}
        
        # Get user details
        user_info = self.recommender.users[self.recommender.users["user_id"] == user_id].iloc[0]
        current_value = user_info[user_feature]
        
        # Create a modified copy of the users dataframe
        modified_users = self.recommender.users.copy()
        modified_users.loc[modified_users["user_id"] == user_id, user_feature] = target_value
        
        # Store original users
        original_users = self.recommender.users
        
        # Set modified users
        self.recommender.users = modified_users
        
        # Get counterfactual recommendations
        counterfactual_recommendations = self.recommender.recommend(
            user_id, n=20, exclude_visited=True
        )
        
        # Reset original users
        self.recommender.users = original_users
        
        # Find new rank of the destination
        new_rank = None
        for i, rec in enumerate(counterfactual_recommendations):
            if rec["destination_id"] == dest_id:
                new_rank = i + 1
                break
        
        if new_rank is None:
            new_rank = "Not in top 20"
        
        # Create explanation
        explanation = {
            "user_id": user_id,
            "destination_id": dest_id,
            "destination_name": current_recommendations[current_rank-1]["name"] if current_rank <= len(current_recommendations) else "Unknown",
            "user_feature": user_feature,
            "current_value": current_value,
            "counterfactual_value": target_value,
            "current_rank": current_rank,
            "counterfactual_rank": new_rank,
            "rank_change": current_rank - new_rank if new_rank != "Not in top 20" else "Dropped out"
        }
        
        return explanation