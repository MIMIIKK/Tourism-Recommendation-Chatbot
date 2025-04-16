import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationExplainer:
    """Class for generating explanations for recommendations"""
    
    def __init__(self, destinations: pd.DataFrame = None, activities: pd.DataFrame = None):
        self.destinations = destinations
        self.activities = activities
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load data if not provided"""
        if self.destinations is None:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
        
        if self.activities is None:
            self.activities = pd.read_pickle(f"{processed_dir}/activities.pkl")
    
    def explain_sustainability(self, destination_id: int) -> Dict[str, Any]:
        """Generate an explanation of a destination's sustainability attributes"""
        if self.destinations is None:
            self.load_data()
        
        # Get destination data
        dest = self.destinations[self.destinations["destination_id"] == destination_id]
        
        if len(dest) == 0:
            return {"error": f"Destination ID {destination_id} not found"}
        
        dest = dest.iloc[0]
        
        # Extract sustainability metrics
        metrics = {
            "Carbon Footprint": dest["carbon_footprint_score"],
            "Water Management": dest["water_consumption_score"],
            "Waste Management": dest["waste_management_score"],
            "Biodiversity Impact": dest["biodiversity_impact_score"],
            "Local Economy Support": dest["local_economy_support_score"]
        }
        
        # Get destination country average (simulated)
        country = dest["country"]
        country_avgs = self.destinations[self.destinations["country"] == country]
        
        if len(country_avgs) > 1:  # Only calculate if there are multiple destinations in country
            country_metrics = {
                "Carbon Footprint": country_avgs["carbon_footprint_score"].mean(),
                "Water Management": country_avgs["water_consumption_score"].mean(),
                "Waste Management": country_avgs["waste_management_score"].mean(),
                "Biodiversity Impact": country_avgs["biodiversity_impact_score"].mean(),
                "Local Economy Support": country_avgs["local_economy_support_score"].mean()
            }
        else:
            country_metrics = None
        
        # Get global average
        global_metrics = {
            "Carbon Footprint": self.destinations["carbon_footprint_score"].mean(),
            "Water Management": self.destinations["water_consumption_score"].mean(),
            "Waste Management": self.destinations["waste_management_score"].mean(),
            "Biodiversity Impact": self.destinations["biodiversity_impact_score"].mean(),
            "Local Economy Support": self.destinations["local_economy_support_score"].mean()
        }
        
        # Get sustainable transportation options
        transportation = dest["sustainable_transportation_options"] if "sustainable_transportation_options" in dest else []
        
        # Get eco certifications
        certifications = dest["eco_certifications"] if "eco_certifications" in dest else "None"
        if isinstance(certifications, str):
            certifications = certifications.split(",") if "," in certifications else [certifications]
        
        # Generate text explanation
        strengths = [k for k, v in metrics.items() if v >= 7.5]
        weaknesses = [k for k, v in metrics.items() if v <= 5.0]
        
        explanation = {
            "destination_name": dest["name"],
            "country": country,
            "overall_score": dest["overall_sustainability_score"],
            "metrics": metrics,
            "country_metrics": country_metrics,
            "global_metrics": global_metrics,
            "transportation_options": transportation,
            "certifications": certifications,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
        return explanation
    
    def explain_recommendation_sources(self, recommendation: Dict[str, Any]) -> str:
        """Explain the sources of a recommendation (for hybrid recommender)"""
        if "sources" not in recommendation:
            return "No source information available for this recommendation."
        
        sources = recommendation["sources"]
        sources_str = ", ".join(sources)
        
        return f"This recommendation comes from the following recommendation methods: {sources_str}"
    
    def generate_sustainability_comparison(self, destination_ids: List[int], save_path: str = None):
        """Generate a visual comparison of sustainability metrics for multiple destinations"""
        if self.destinations is None:
            self.load_data()
        
        # Filter destinations
        dests = self.destinations[self.destinations["destination_id"].isin(destination_ids)]
        
        if len(dests) == 0:
            return None
        
        # Prepare data for plotting
        metrics = [
            "carbon_footprint_score", 
            "water_consumption_score",
            "waste_management_score", 
            "biodiversity_impact_score",
            "local_economy_support_score"
        ]
        
        metric_names = [
            "Carbon Footprint", 
            "Water Management",
            "Waste Management", 
            "Biodiversity Impact",
            "Local Economy"
        ]
        
        # Create a dataframe for plotting
        plot_data = []
        for _, dest in dests.iterrows():
            for i, metric in enumerate(metrics):
                plot_data.append({
                    "Destination": dest["name"],
                    "Metric": metric_names[i],
                    "Score": dest[metric]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Metric", y="Score", hue="Destination", data=plot_df)
        plt.title("Sustainability Metrics Comparison")
        plt.ylim(0, 10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            return plt
    
    def generate_counterfactual_explanation(self, destination_id: int) -> Dict[str, Any]:
        """Generate a counterfactual explanation showing how a destination could improve"""
        if self.destinations is None:
            self.load_data()
        
        dest = self.destinations[self.destinations["destination_id"] == destination_id]
        if len(dest) == 0:
            return {"error": f"Destination ID {destination_id} not found"}
        
        dest = dest.iloc[0]
        
        # Find similar destinations with better sustainability
        country = dest["country"]
        landscape = dest["landscape_type"]
        current_score = dest["overall_sustainability_score"]
        
        # Find destinations with similar attributes but better sustainability
        similar_better = self.destinations[
            (self.destinations["country"] == country) &
            (self.destinations["landscape_type"] == landscape) &
            (self.destinations["overall_sustainability_score"] > current_score)
        ]
        
        if len(similar_better) == 0:
            # If no better destination in same country and landscape, look more broadly
            similar_better = self.destinations[
                (self.destinations["landscape_type"] == landscape) &
                (self.destinations["overall_sustainability_score"] > current_score)
            ]
        
        if len(similar_better) == 0:
            return {"message": f"No similar destinations with better sustainability scores found"}
        
        # Sort by sustainability score and take the best example
        similar_better = similar_better.sort_values("overall_sustainability_score", ascending=False)
        better_example = similar_better.iloc[0]
        
        # Calculate differences in metrics
        metrics = [
            "carbon_footprint_score", 
            "water_consumption_score",
            "waste_management_score", 
            "biodiversity_impact_score",
            "local_economy_support_score"
        ]
        
        metric_names = [
            "Carbon Footprint", 
            "Water Management",
            "Waste Management", 
            "Biodiversity Impact",
            "Local Economy Support"
        ]
        
        differences = {}
        improvement_areas = []
        
        for i, metric in enumerate(metrics):
            diff = better_example[metric] - dest[metric]
            differences[metric_names[i]] = round(diff, 1)
            
            if diff >= 2:  # Significant difference
                improvement_areas.append(metric_names[i])
        
        return {
            "current_destination": dest["name"],
            "better_destination": better_example["name"],
            "current_score": current_score,
            "better_score": better_example["overall_sustainability_score"],
            "score_difference": round(better_example["overall_sustainability_score"] - current_score, 1),
            "metric_differences": differences,
            "improvement_areas": improvement_areas
        }