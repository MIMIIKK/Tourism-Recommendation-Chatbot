import numpy as np
from typing import Dict, List, Any, Callable

class SustainabilityWeighting:
    """
    Class for defining and applying different sustainability weighting schemes
    to recommendation scores
    """
    
    def __init__(self):
        # Define standard weighting functions
        self.weighting_functions = {
            "linear": self._linear_weighting,
            "quadratic": self._quadratic_weighting,
            "sigmoid": self._sigmoid_weighting,
            "threshold": self._threshold_weighting
        }
    
    def _linear_weighting(self, base_score: float, sustainability_score: float, 
                         weight: float = 0.5) -> float:
        """
        Linear weighting: weighted average of base score and sustainability score
        
        Parameters:
        - base_score: Original recommendation score (0-1)
        - sustainability_score: Sustainability score (0-1)
        - weight: Weight for sustainability (0-1)
        
        Returns:
        - Weighted score (0-1)
        """
        return (1 - weight) * base_score + weight * sustainability_score
    
    def _quadratic_weighting(self, base_score: float, sustainability_score: float, 
                            weight: float = 0.5) -> float:
        """
        Quadratic weighting: gives more emphasis to higher sustainability scores
        
        Parameters:
        - base_score: Original recommendation score (0-1)
        - sustainability_score: Sustainability score (0-1)
        - weight: Weight for sustainability (0-1)
        
        Returns:
        - Weighted score (0-1)
        """
        # Square the sustainability score to emphasize higher values
        sustainability_factor = sustainability_score ** 2
        return (1 - weight) * base_score + weight * sustainability_factor
    
    def _sigmoid_weighting(self, base_score: float, sustainability_score: float, 
                          weight: float = 0.5) -> float:
        """
        Sigmoid weighting: creates a softer threshold effect
        
        Parameters:
        - base_score: Original recommendation score (0-1)
        - sustainability_score: Sustainability score (0-1)
        - weight: Weight for sustainability (0-1)
        
        Returns:
        - Weighted score (0-1)
        """
        # Apply sigmoid transformation to sustainability score
        # This creates a soft threshold around 0.5
        def sigmoid(x):
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
        
        sustainability_factor = sigmoid(sustainability_score)
        return (1 - weight) * base_score + weight * sustainability_factor
    
    def _threshold_weighting(self, base_score: float, sustainability_score: float, 
                            weight: float = 0.5, threshold: float = 0.7) -> float:
        """
        Threshold weighting: penalizes items below a sustainability threshold
        
        Parameters:
        - base_score: Original recommendation score (0-1)
        - sustainability_score: Sustainability score (0-1)
        - weight: Weight for sustainability (0-1)
        - threshold: Sustainability threshold (0-1)
        
        Returns:
        - Weighted score (0-1)
        """
        # Apply a penalty to items below the threshold
        if sustainability_score < threshold:
            penalty = (threshold - sustainability_score) / threshold
            sustainability_factor = sustainability_score * (1 - penalty * weight)
        else:
            sustainability_factor = sustainability_score
        
        return (1 - weight) * base_score + weight * sustainability_factor
    
    def apply_weighting(self, base_scores: List[float], sustainability_scores: List[float], 
                       scheme: str = "linear", weight: float = 0.5, 
                       **kwargs) -> List[float]:
        """
        Apply a sustainability weighting scheme to a list of base scores
        
        Parameters:
        - base_scores: List of original recommendation scores (0-1)
        - sustainability_scores: List of sustainability scores (0-1)
        - scheme: Weighting scheme to use ("linear", "quadratic", "sigmoid", "threshold")
        - weight: Weight for sustainability (0-1)
        - kwargs: Additional parameters for specific weighting schemes
        
        Returns:
        - List of weighted scores (0-1)
        """
        if scheme not in self.weighting_functions:
            raise ValueError(f"Unknown weighting scheme: {scheme}. Available schemes: {list(self.weighting_functions.keys())}")
        
        weighting_function = self.weighting_functions[scheme]
        
        weighted_scores = []
        for base, sust in zip(base_scores, sustainability_scores):
            weighted = weighting_function(base, sust, weight, **kwargs)
            weighted_scores.append(weighted)
        
        return weighted_scores
    
    def apply_weighting_to_recommendations(self, recommendations: List[Dict[str, Any]], 
                                          scheme: str = "linear", weight: float = 0.5, 
                                          **kwargs) -> List[Dict[str, Any]]:
        """
        Apply a sustainability weighting scheme to a list of recommendations
        
        Parameters:
        - recommendations: List of recommendation dictionaries
        - scheme: Weighting scheme to use
        - weight: Weight for sustainability (0-1)
        - kwargs: Additional parameters for specific weighting schemes
        
        Returns:
        - Reordered list of recommendations
        """
        if not recommendations:
            return []
        
        # Extract base scores (based on original order)
        n_recs = len(recommendations)
        base_scores = [(n_recs - i) / n_recs for i in range(n_recs)]
        
        # Extract sustainability scores
        sustainability_scores = [
            rec["sustainability_score"] / 10 if "sustainability_score" in rec else 0.5 
            for rec in recommendations
        ]
        
        # Apply weighting
        weighted_scores = self.apply_weighting(base_scores, sustainability_scores, scheme, weight, **kwargs)
        
        # Combine recommendations with weighted scores
        weighted_recs = []
        for i, rec in enumerate(recommendations):
            weighted_recs.append({
                "recommendation": rec,
                "base_score": base_scores[i],
                "sustainability_score": sustainability_scores[i],
                "weighted_score": weighted_scores[i]
            })
        
        # Sort by weighted score
        weighted_recs = sorted(weighted_recs, key=lambda x: x["weighted_score"], reverse=True)
        
        # Return reordered recommendations
        return [item["recommendation"] for item in weighted_recs]
    
    def register_custom_weighting(self, name: str, function: Callable) -> None:
        """
        Register a custom weighting function
        
        Parameters:
        - name: Name of the weighting scheme
        - function: Weighting function with signature (base_score, sustainability_score, weight, **kwargs) -> float
        """
        self.weighting_functions[name] = function