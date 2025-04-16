import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from src.models.base_models import BaseRecommender, PopularityRecommender, ContentBasedRecommender, CollaborativeFilteringRecommender
from src.models.neural_cf import NeuralCollaborativeFiltering
from src.sustainability.sustainability_scorer import SustainabilityScorer

class HybridRecommender(BaseRecommender):
    """Hybrid recommendation model combining multiple recommenders"""
    
    def __init__(self, recommenders: List[BaseRecommender] = None, 
                 weights: List[float] = None, 
                 sustainability_weight: float = 0.3):
        super().__init__(name="HybridRecommender")
        self.recommenders = recommenders or []
        self.weights = weights or []
        self.sustainability_weight = sustainability_weight
        self.sustainability_scorer = None
    
    def add_recommender(self, recommender: BaseRecommender, weight: float = 1.0):
        """Add a recommender to the hybrid model"""
        self.recommenders.append(recommender)
        self.weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        return self
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load data for all recommenders"""
        super().load_data(processed_dir)
        
        for recommender in self.recommenders:
            recommender.load_data(processed_dir)
        
        # Initialize sustainability scorer
        self.sustainability_scorer = SustainabilityScorer(self.destinations)
    
    def fit(self):
        """Train all recommenders"""
        for recommender in self.recommenders:
            print(f"Training {recommender.name}...")
            recommender.fit()
        
        return self
    
    def recommend(self, user_id: int, n: int = 5, exclude_visited: bool = True) -> List[Dict[str, Any]]:
        """Generate recommendations by combining multiple recommenders"""
        if not self.recommenders:
            raise ValueError("No recommenders added to hybrid model")
        
        # Get recommendations from each recommender
        all_recommendations = {}
        
        for i, recommender in enumerate(self.recommenders):
            recs = recommender.recommend(user_id, n=n*2, exclude_visited=exclude_visited)
            weight = self.weights[i]
            
            for j, rec in enumerate(recs):
                dest_id = rec["destination_id"]
                # Score is based on position and recommender weight
                score = weight * (1.0 - (j / len(recs)))
                
                if dest_id in all_recommendations:
                    all_recommendations[dest_id]["score"] += score
                    all_recommendations[dest_id]["sources"].append(recommender.name)
                else:
                    all_recommendations[dest_id] = {
                        "recommendation": rec,
                        "score": score,
                        "sources": [recommender.name]
                    }
        
        # Sort recommendations by score
        sorted_recommendations = sorted(
            all_recommendations.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # Get top n recommendations
        top_recommendations = [item["recommendation"] for item in sorted_recommendations[:n]]
        
        # Apply sustainability weighting
        if self.sustainability_weight > 0:
            top_recommendations = self.sustainability_scorer.apply_sustainability_weighting(
                top_recommendations, 
                weight=self.sustainability_weight
            )
        
        return top_recommendations


def create_default_hybrid_recommender() -> HybridRecommender:
    """Create a default hybrid recommender with standard components"""
    # Create individual recommenders
    popularity_rec = PopularityRecommender()
    content_rec = ContentBasedRecommender()
    user_cf_rec = CollaborativeFilteringRecommender(method="user")
    item_cf_rec = CollaborativeFilteringRecommender(method="item")
    
    # Create hybrid recommender
    hybrid = HybridRecommender()
    
    # Add recommenders with weights
    hybrid.add_recommender(popularity_rec, weight=0.1)
    hybrid.add_recommender(content_rec, weight=0.3)
    hybrid.add_recommender(user_cf_rec, weight=0.3)
    hybrid.add_recommender(item_cf_rec, weight=0.3)
    
    return hybrid