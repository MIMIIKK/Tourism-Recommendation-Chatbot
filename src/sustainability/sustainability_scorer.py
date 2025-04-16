import numpy as np
import pandas as pd
from typing import Dict, List, Any

class SustainabilityScorer:
    """Class for calculating and applying sustainability scores to recommendations"""
    
    def __init__(self, destinations: pd.DataFrame = None):
        self.destinations = destinations
        self.sustainability_weights = {
            "carbon_footprint_score": 0.25,
            "water_consumption_score": 0.20,
            "waste_management_score": 0.20,
            "biodiversity_impact_score": 0.20,
            "local_economy_support_score": 0.15
        }
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load destination data if not provided"""
        if self.destinations is None:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
    
    def get_sustainability_score(self, destination_id: int) -> float:
        """Get the overall sustainability score for a destination"""
        dest = self.destinations[self.destinations["destination_id"] == destination_id]
        if len(dest) == 0:
            return 0.0
        
        return dest.iloc[0]["overall_sustainability_score"]
    
    def get_detailed_sustainability_scores(self, destination_id: int) -> Dict[str, float]:
        """Get detailed sustainability scores for a destination"""
        dest = self.destinations[self.destinations["destination_id"] == destination_id]
        if len(dest) == 0:
            return {}
        
        scores = {}
        for metric in self.sustainability_weights.keys():
            scores[metric] = dest.iloc[0][metric]
        
        scores["overall"] = dest.iloc[0]["overall_sustainability_score"]
        return scores
    
    def apply_sustainability_weighting(self, recommendations: List[Dict[str, Any]], 
                                       weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply sustainability weighting to recommendations
        
        Parameters:
        - recommendations: List of recommendation dictionaries
        - weight: Weight of sustainability score (0-1), where 0 means no impact and 
                 1 means recommendations are based solely on sustainability
        
        Returns:
        - Reordered recommendations with sustainability weighting
        """
        # Ensure destinations data is loaded
        if self.destinations is None:
            self.load_data()
        
        # Calculate weighted scores
        weighted_recs = []
        
        # Normalize original recommendation order to get base scores
        # (first recommendation = 1.0, last recommendation = 0.0)
        n_recs = len(recommendations)
        
        for i, rec in enumerate(recommendations):
            dest_id = rec["destination_id"]
            
            # Base score from original recommendation order (inverted position / total)
            base_score = (n_recs - i) / n_recs if n_recs > 1 else 1.0
            
            # Get sustainability score (normalized to 0-1 scale)
            sustainability_score = self.get_sustainability_score(dest_id) / 10.0
            
            # Calculate weighted score
            weighted_score = (1 - weight) * base_score + weight * sustainability_score
            
            weighted_recs.append({
                "recommendation": rec,
                "weighted_score": weighted_score,
                "base_score": base_score,
                "sustainability_score": sustainability_score
            })
        
        # Sort by weighted score
        weighted_recs = sorted(weighted_recs, key=lambda x: x["weighted_score"], reverse=True)
        
        # Return reordered recommendations
        return [item["recommendation"] for item in weighted_recs]
    
    def filter_by_sustainability_threshold(self, recommendations: List[Dict[str, Any]], 
                                          threshold: float = 6.0) -> List[Dict[str, Any]]:
        """Filter recommendations to include only those above a sustainability threshold"""
        if self.destinations is None:
            self.load_data()
        
        filtered_recs = []
        
        for rec in recommendations:
            dest_id = rec["destination_id"]
            sustainability_score = self.get_sustainability_score(dest_id)
            
            if sustainability_score >= threshold:
                filtered_recs.append(rec)
        
        return filtered_recs