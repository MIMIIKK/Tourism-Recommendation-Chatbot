import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class BaseRecommender:
    """Base class for recommendation models"""
    
    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.destinations = None
        self.users = None
        self.interaction_matrix = None
        self.user_ids = None
        self.dest_ids = None
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load processed data"""
        self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
        self.users = pd.read_pickle(f"{processed_dir}/users.pkl")
        
        with open(f"{processed_dir}/interaction_matrix.pkl", "rb") as f:
            data = pickle.load(f)
            self.interaction_matrix = data["matrix"]
            self.user_ids = data["user_ids"]
            self.dest_ids = data["dest_ids"]
    
    def fit(self, *args, **kwargs):
        """Train the model - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def recommend(self, user_id: int, n: int = 5, *args, **kwargs) -> List[Dict[str, Any]]:
        """Generate recommendations - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement recommend()")
    
    def _get_user_index(self, user_id: int) -> int:
        """Convert user_id to matrix index"""
        return np.where(self.user_ids == user_id)[0][0]
    
    def _get_destination_index(self, dest_id: int) -> int:
        """Convert destination_id to matrix index"""
        return np.where(self.dest_ids == dest_id)[0][0]
    
    def _get_user_id(self, user_index: int) -> int:
        """Convert matrix index to user_id"""
        return self.user_ids[user_index]
    
    def _get_destination_id(self, dest_index: int) -> int:
        """Convert matrix index to destination_id"""
        return self.dest_ids[dest_index]
    
    def _format_recommendations(self, dest_indices: List[int]) -> List[Dict[str, Any]]:
        """Format recommendations as a list of dictionaries with destination details"""
        recommendations = []
        
        for idx in dest_indices:
            dest_id = self._get_destination_id(idx)
            dest_info = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
            
            recommendations.append({
                "destination_id": int(dest_id),
                "name": dest_info["name"],
                "country": dest_info["country"],
                "sustainability_score": dest_info["overall_sustainability_score"],
                "landscape": dest_info["landscape_type"]
            })
        
        return recommendations


class PopularityRecommender(BaseRecommender):
    """Recommends the most popular destinations"""
    
    def __init__(self):
        super().__init__(name="PopularityRecommender")
        self.popularity_scores = None
    
    def fit(self):
        """Calculate popularity scores for all destinations"""
        # Popularity is sum of interactions (visits) for each destination
        self.popularity_scores = np.sum(self.interaction_matrix, axis=0)
        return self
    
    def recommend(self, user_id: int, n: int = 5, exclude_visited: bool = True) -> List[Dict[str, Any]]:
        """Recommend the most popular destinations"""
        user_idx = self._get_user_index(user_id)
        
        # Get indices of destinations sorted by popularity
        dest_indices = np.argsort(-self.popularity_scores)
        
        if exclude_visited:
            # Filter out destinations the user has already visited
            visited = self.interaction_matrix[user_idx] > 0
            dest_indices = dest_indices[~visited[dest_indices]]
        
        # Return top n recommendations
        top_n_indices = dest_indices[:n]
        return self._format_recommendations(top_n_indices)


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation model using destination features"""
    
    def __init__(self):
        super().__init__(name="ContentBasedRecommender")
        self.destination_features = None
        self.feature_similarity = None
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load processed data including destination features"""
        super().load_data(processed_dir)
        
        with open(f"{processed_dir}/destination_features.pkl", "rb") as f:
            data = pickle.load(f)
            self.destination_features = data["features"]
    
    def fit(self):
        """Calculate similarity between destinations based on features"""
        self.feature_similarity = cosine_similarity(self.destination_features)
        return self
    
    def recommend(self, user_id: int, n: int = 5, exclude_visited: bool = True) -> List[Dict[str, Any]]:
        """Recommend destinations similar to those the user has visited"""
        user_idx = self._get_user_index(user_id)
        
        # Get destinations the user has visited
        visited_indices = np.where(self.interaction_matrix[user_idx] > 0)[0]
        
        if len(visited_indices) == 0:
            # If user has no history, fall back to popularity-based recommendations
            recommender = PopularityRecommender()
            recommender.load_data()
            recommender.fit()
            return recommender.recommend(user_id, n, exclude_visited)
        
        # Calculate average similarity to visited destinations for each destination
        similarity_scores = np.zeros(len(self.dest_ids))
        
        for idx in visited_indices:
            similarity_scores += self.feature_similarity[idx]
        
        # Average the scores
        similarity_scores /= len(visited_indices)
        
        # Sort destinations by similarity score
        dest_indices = np.argsort(-similarity_scores)
        
        if exclude_visited:
            # Filter out destinations the user has already visited
            visited = self.interaction_matrix[user_idx] > 0
            dest_indices = dest_indices[~visited[dest_indices]]
        
        # Return top n recommendations
        top_n_indices = dest_indices[:n]
        return self._format_recommendations(top_n_indices)


class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering recommendation model"""
    
    def __init__(self, method: str = "user"):
        super().__init__(name=f"{method.capitalize()}BasedCF")
        self.method = method  # "user" or "item"
        self.similarity_matrix = None
    
    def fit(self):
        """Calculate similarity matrix based on the chosen method"""
        if self.method == "user":
            # User-based CF: Calculate similarity between users
            self.similarity_matrix = cosine_similarity(self.interaction_matrix)
        elif self.method == "item":
            # Item-based CF: Calculate similarity between items (destinations)
            self.similarity_matrix = cosine_similarity(self.interaction_matrix.T)
        else:
            raise ValueError("Method must be either 'user' or 'item'")
        
        return self
    
    def recommend(self, user_id: int, n: int = 5, exclude_visited: bool = True) -> List[Dict[str, Any]]:
        """Generate recommendations based on collaborative filtering"""
        user_idx = self._get_user_index(user_id)
        
        if self.method == "user":
            # User-based CF
            # Get similarity scores for the target user with all other users
            user_similarities = self.similarity_matrix[user_idx]
            
            # Exclude self-similarity
            user_similarities[user_idx] = 0
            
            # Get top similar users
            similar_users = np.argsort(-user_similarities)[:20]  # Consider top 20 similar users
            
            # Calculate recommendation scores
            scores = np.zeros(self.interaction_matrix.shape[1])
            
            for similar_user_idx in similar_users:
                # Skip users with zero similarity
                if user_similarities[similar_user_idx] <= 0:
                    continue
                
                # Add weighted scores from similar users
                scores += user_similarities[similar_user_idx] * self.interaction_matrix[similar_user_idx]
            
            # Normalize scores
            if np.sum(scores) > 0:
                scores = scores / np.sum(scores)
        
        elif self.method == "item":
            # Item-based CF
            # Get user's interaction history
            user_interactions = self.interaction_matrix[user_idx]
            
            # Calculate weighted scores based on item similarity
            scores = np.zeros(len(self.dest_ids))
            
            for i, has_visited in enumerate(user_interactions):
                if has_visited > 0:
                    # Add similarity scores of this visited destination to all others
                    similarities = self.similarity_matrix[i]
                    scores += similarities
            
            # Normalize scores
            if np.sum(scores) > 0:
                scores = scores / np.sum(scores)
        
        # Sort destinations by score
        dest_indices = np.argsort(-scores)
        
        if exclude_visited:
            # Filter out destinations the user has already visited
            visited = self.interaction_matrix[user_idx] > 0
            dest_indices = dest_indices[~visited[dest_indices]]
        
        # Return top n recommendations
        top_n_indices = dest_indices[:n]
        return self._format_recommendations(top_n_indices)