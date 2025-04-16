import pandas as pd
import numpy as np
from typing import Dict, List, Any

class EnvironmentalImpactCalculator:
    """Calculate environmental impact of tourism choices"""
    
    def __init__(self, destinations: pd.DataFrame = None, activities: pd.DataFrame = None):
        self.destinations = destinations
        self.activities = activities
        
        # Define impact weights
        self.impact_weights = {
            "carbon": 0.4,
            "water": 0.2,
            "waste": 0.2,
            "biodiversity": 0.2
        }
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load data if not provided"""
        if self.destinations is None:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
        
        if self.activities is None:
            self.activities = pd.read_pickle(f"{processed_dir}/activities.pkl")
    
    def calculate_destination_impact(self, destination_id: int) -> Dict[str, float]:
        """Calculate environmental impact of a destination"""
        if self.destinations is None:
            self.load_data()
        
        dest = self.destinations[self.destinations["destination_id"] == destination_id]
        
        if len(dest) == 0:
            return {"error": f"Destination ID {destination_id} not found"}
        
        dest = dest.iloc[0]
        
        # Extract impact metrics (we invert the scores as higher score means less impact)
        carbon_impact = 10 - dest["carbon_footprint_score"]
        water_impact = 10 - dest["water_consumption_score"]
        waste_impact = 10 - dest["waste_management_score"]
        biodiversity_impact = 10 - dest["biodiversity_impact_score"]
        
        # Calculate weighted impact score
        total_impact = (carbon_impact * self.impact_weights["carbon"] + 
                        water_impact * self.impact_weights["water"] + 
                        waste_impact * self.impact_weights["waste"] + 
                        biodiversity_impact * self.impact_weights["biodiversity"])
        
        # Scale to 0-100 for better interpretation (0 = no impact, 100 = maximum impact)
        impact_percentage = total_impact * 10
        
        return {
            "carbon_impact": carbon_impact,
            "water_impact": water_impact,
            "waste_impact": waste_impact,
            "biodiversity_impact": biodiversity_impact,
            "total_impact_score": total_impact,
            "impact_percentage": impact_percentage
        }
    
    def calculate_activity_impact(self, activity_id: int) -> Dict[str, float]:
        """Calculate environmental impact of an activity"""
        if self.activities is None:
            self.load_data()
        
        activity = self.activities[self.activities["activity_id"] == activity_id]
        
        if len(activity) == 0:
            return {"error": f"Activity ID {activity_id} not found"}
        
        activity = activity.iloc[0]
        
        # Extract impact metrics (we invert as higher score means less impact)
        environmental_impact = activity["environmental_impact_score"]
        resource_consumption = activity["resource_consumption_score"]
        
        # Calculate positive impact (benefit)
        community_benefit = activity["local_community_benefit_score"]
        
        # Calculate net impact (impact minus benefit)
        net_impact = (environmental_impact + resource_consumption - community_benefit) / 2
        
        # Scale net impact to 0-10 scale
        scaled_impact = min(max(net_impact, 0), 10)
        
        return {
            "environmental_impact": environmental_impact,
            "resource_consumption": resource_consumption,
            "community_benefit": community_benefit,
            "net_impact": net_impact,
            "scaled_impact": scaled_impact
        }
    
    def calculate_itinerary_impact(self, destination_id: int, activity_ids: List[int]) -> Dict[str, Any]:
        """Calculate total environmental impact of an itinerary"""
        # Calculate destination impact
        dest_impact = self.calculate_destination_impact(destination_id)
        
        if "error" in dest_impact:
            return dest_impact
        
        # Calculate activity impacts
        activity_impacts = []
        total_activity_impact = 0
        
        for act_id in activity_ids:
            impact = self.calculate_activity_impact(act_id)
            
            if "error" not in impact:
                activity_impacts.append({
                    "activity_id": act_id,
                    "impact": impact
                })
                total_activity_impact += impact["scaled_impact"]
        
        # Calculate average activity impact
        avg_activity_impact = total_activity_impact / len(activity_ids) if activity_ids else 0
        
        # Calculate combined impact (60% destination, 40% activities)
        combined_impact = dest_impact["impact_percentage"] * 0.6 + avg_activity_impact * 4  # Scale activity impact to 0-100
        
        return {
            "destination_impact": dest_impact,
            "activity_impacts": activity_impacts,
            "average_activity_impact": avg_activity_impact,
            "combined_impact": combined_impact,
            "impact_category": self._categorize_impact(combined_impact)
        }
    
    def _categorize_impact(self, impact_score: float) -> str:
        """Categorize impact score into text category"""
        if impact_score < 20:
            return "Very Low Impact"
        elif impact_score < 40:
            return "Low Impact"
        elif impact_score < 60:
            return "Moderate Impact"
        elif impact_score < 80:
            return "High Impact"
        else:
            return "Very High Impact"
    
    def compare_destinations(self, destination_ids: List[int]) -> Dict[str, Any]:
        """Compare environmental impact of multiple destinations"""
        impacts = {}
        
        for dest_id in destination_ids:
            impact = self.calculate_destination_impact(dest_id)
            if "error" not in impact:
                dest = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
                impacts[dest_id] = {
                    "name": dest["name"],
                    "country": dest["country"],
                    "impact": impact
                }
        
        # Sort destinations by impact (low to high)
        sorted_impacts = sorted(impacts.items(), key=lambda x: x[1]["impact"]["impact_percentage"])
        
        # Format results
        return {
            "impacts": impacts,
            "sorted": sorted_impacts,
            "best_option": sorted_impacts[0] if sorted_impacts else None,
            "worst_option": sorted_impacts[-1] if sorted_impacts else None
        }