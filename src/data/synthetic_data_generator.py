import pandas as pd
import numpy as np
import random
import os
from typing import Dict, List, Tuple

class SyntheticDataGenerator:
    def __init__(self, 
                 num_destinations: int = 100, 
                 num_activities: int = 300, 
                 num_users: int = 1000):
        self.num_destinations = num_destinations
        self.num_activities = num_activities
        self.num_users = num_users

        
        self.countries = ["Spain", "France", "Italy", "Greece", "Portugal", "Japan", 
                          "Thailand", "Indonesia", "Costa Rica", "New Zealand", 
                          "Canada", "Norway", "Sweden", "Iceland", "Finland"]
        self.landscapes = ["Beach", "Mountain", "City", "Forest", "Desert", 
                           "Island", "Countryside", "Lake", "River", "Coastal"]
        self.activities = ["Hiking", "Swimming", "Cultural Tours", "Wildlife Watching", 
                           "Cycling", "Kayaking", "Surfing", "Local Cuisine", "Museums", 
                           "Historical Sites", "Eco Tours", "Volunteering"]
        
    def generate_destinations(self) -> pd.DataFrame:
        """Generate synthetic destination data"""
        destinations = []
        
        for i in range(1, self.num_destinations + 1):
            country = random.choice(self.countries)
            landscape = random.choice(self.landscapes)
            activities = random.sample(self.activities, k=random.randint(3, 8))
            
            destination = {
                "destination_id": i,
                "name": f"Destination {i}",
                "country": country,
                "region": f"Region {random.randint(1, 5)} in {country}",
                "climate": random.choice(["Tropical", "Mediterranean", "Continental", "Polar", "Arid"]),
                "landscape_type": landscape,
                "popular_activities": ",".join(activities),
                "accommodation_types": ",".join(random.sample(["Hotel", "Hostel", "Resort", "Eco-lodge", "Apartment", "Camping"], 
                                                              k=random.randint(2, 4))),
                "peak_season": random.choice(["Summer", "Winter", "Spring", "Fall", "Year-round"]),
                "off_peak_season": random.choice(["Summer", "Winter", "Spring", "Fall"])
            }
            destinations.append(destination)
        
        return pd.DataFrame(destinations)
    
    def generate_sustainability_metrics(self, destination_ids: List[int]) -> pd.DataFrame:
        """Generate synthetic sustainability metrics for destinations"""
        metrics = []
        
        for dest_id in destination_ids:
            # Create base scores with some correlation
            base_eco_score = random.uniform(3, 10)
            # Add some variation but keep correlation
            carbon_score = max(1, min(10, base_eco_score + random.uniform(-2, 2)))
            water_score = max(1, min(10, base_eco_score + random.uniform(-2, 2)))
            waste_score = max(1, min(10, base_eco_score + random.uniform(-2, 2)))
            biodiversity_score = max(1, min(10, base_eco_score + random.uniform(-2, 2)))
            local_economy_score = max(1, min(10, base_eco_score + random.uniform(-2, 2)))
            
            # Calculate overall score with weighted components
            overall_score = (carbon_score * 0.25 + 
                            water_score * 0.2 + 
                            waste_score * 0.2 + 
                            biodiversity_score * 0.2 + 
                            local_economy_score * 0.15)
            
            # Generate eco certifications based on overall score
            possible_certifications = ["Green Globe", "EarthCheck", "LEED", "Blue Flag", "Green Key"]
            
            # Higher scoring destinations get more certifications
            num_certifications = int(overall_score / 3)
            certifications = random.sample(possible_certifications, 
                                          k=min(num_certifications, len(possible_certifications)))
            
            metric = {
                "destination_id": dest_id,
                "carbon_footprint_score": round(carbon_score, 1),
                "water_consumption_score": round(water_score, 1),
                "waste_management_score": round(waste_score, 1),
                "biodiversity_impact_score": round(biodiversity_score, 1),
                "local_economy_support_score": round(local_economy_score, 1),
                "conservation_initiatives": f"{random.randint(0, 5)} active conservation projects",
                "eco_certifications": ",".join(certifications) if certifications else "None",
                "sustainable_transportation_options": ",".join(
                    random.sample(["Public Transit", "Bike Sharing", "Electric Vehicles", "Walking Paths"], 
                                 k=random.randint(0, 4))),
                "overall_sustainability_score": round(overall_score, 2)
            }
            
            metrics.append(metric)
        
        return pd.DataFrame(metrics)
    
    def generate_activities(self, destination_ids: List[int]) -> pd.DataFrame:
        """Generate synthetic activities data"""
        activities = []
        activity_id = 1
        
        activity_categories = ["Adventure", "Cultural", "Nature", "Relaxation", 
                              "Culinary", "Educational", "Volunteer"]
        
        for dest_id in destination_ids:
            # Number of activities per destination varies
            num_activities_for_dest = random.randint(2, 6)
            
            for _ in range(num_activities_for_dest):
                category = random.choice(activity_categories)
                
                # Activities in nature and volunteer categories tend to be more eco-friendly
                if category in ["Nature", "Volunteer"]:
                    env_impact = random.uniform(1, 5)
                    community_benefit = random.uniform(6, 10)
                elif category == "Adventure":
                    env_impact = random.uniform(4, 8)
                    community_benefit = random.uniform(3, 8)
                else:
                    env_impact = random.uniform(3, 7)
                    community_benefit = random.uniform(4, 9)
                
                activity = {
                    "activity_id": activity_id,
                    "name": f"{category} Activity {activity_id}",
                    "description": f"This is a {category.lower()} activity in destination {dest_id}",
                    "destination_id": dest_id,
                    "category": category,
                    "environmental_impact_score": round(env_impact, 1),
                    "local_community_benefit_score": round(community_benefit, 1),
                    "resource_consumption_score": round(random.uniform(1, 10), 1),
                    "seasonal_availability": random.choice(["Year-round", "Summer only", "Winter only", 
                                                          "Spring/Fall", "Seasonal"])
                }
                
                activities.append(activity)
                activity_id += 1
        
        return pd.DataFrame(activities)
    
    def generate_users(self) -> pd.DataFrame:
        """Generate synthetic user data"""
        users = []
        
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        interests = ["Nature", "Culture", "Adventure", "Relaxation", "Food", 
                     "Photography", "History", "Wildlife", "Beach", "Mountains"]
        travel_styles = ["Eco-conscious", "Luxury", "Budget", "Adventure", 
                         "Family", "Solo", "Cultural"]
        
        for i in range(1, self.num_users + 1):
            # Correlate sustainability preference with travel style
            travel_style = random.choice(travel_styles)
            
            if travel_style == "Eco-conscious":
                sustainability_pref = random.uniform(7, 10)
            elif travel_style in ["Luxury", "Family"]:
                sustainability_pref = random.uniform(4, 8)
            else:
                sustainability_pref = random.uniform(3, 9)
            
            # Generate random travel history
            travel_history = random.sample(range(1, self.num_destinations + 1), 
                                          k=random.randint(0, 10))
            
            user_interests = random.sample(interests, k=random.randint(2, 5))
            
            user = {
                "user_id": i,
                "age_group": random.choice(age_groups),
                "interests": ",".join(user_interests),
                "travel_history": ",".join(str(x) for x in travel_history),
                "sustainability_preference": round(sustainability_pref, 1),
                "budget_level": random.randint(1, 5),
                "preferred_activities": ",".join(random.sample(self.activities, k=random.randint(2, 5))),
                "travel_style": travel_style
            }
            
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets and return them as a dictionary"""
        print("Generating destination data...")
        destinations = self.generate_destinations()
        
        print("Generating sustainability metrics...")
        sustainability = self.generate_sustainability_metrics(destinations["destination_id"].tolist())
        
        print("Generating activities data...")
        activities = self.generate_activities(destinations["destination_id"].tolist())
        
        print("Generating user data...")
        users = self.generate_users()
        
        return {
            "destinations": destinations,
            "sustainability_metrics": sustainability,
            "activities": activities,
            "users": users
        }
    
    def save_data(self, data_dir: str = "data/raw"):
        """Generate and save all datasets to the specified directory"""
        os.makedirs(data_dir, exist_ok=True)
        
        datasets = self.generate_all_data()
        
        for name, df in datasets.items():
            file_path = f"{data_dir}/{name}.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved {name} data to {file_path}")


if __name__ == "__main__":
    # Generate synthetic data
    generator = SyntheticDataGenerator(
        num_destinations=100,
        num_activities=300,
        num_users=1000
    )
    generator.save_data()