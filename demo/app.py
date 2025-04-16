import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Dict, List, Any

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ensemble import create_default_hybrid_recommender
from src.explainability.explainations import RecommendationExplainer  # Fixed typo in import
from src.sustainability.sustainability_scorer import SustainabilityScorer
from src.chatbot.chatbot_interface import SustainableTourismChatbot  # Added chatbot import

class SustainableTourismDemo:
    """Demo application for the Sustainable Tourism Recommender System"""
    
    def __init__(self):
        self.recommender = None
        self.explainer = None
        self.sustainability_scorer = None
        self.chatbot = None  # Added chatbot
        self.users = None
        self.destinations = None
        self.activities = None
    
    def load_data(self, processed_dir: str = "data/processed"):
        """Load processed data"""
        print("Loading data...")
        try:
            self.destinations = pd.read_pickle(f"{processed_dir}/destinations.pkl")
            self.activities = pd.read_pickle(f"{processed_dir}/activities.pkl")
            self.users = pd.read_pickle(f"{processed_dir}/users.pkl")
            
            print(f"Loaded {len(self.destinations)} destinations")
            print(f"Loaded {len(self.activities)} activities")
            print(f"Loaded {len(self.users)} users")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Make sure you've generated and processed the data using 'python main.py generate' and 'python main.py process'")
            sys.exit(1)
    
    def initialize_recommender(self, sustainability_weight: float = 0.3):
        """Initialize the hybrid recommender system"""
        print("Initializing recommender...")
        self.recommender = create_default_hybrid_recommender()
        self.recommender.sustainability_weight = sustainability_weight
        self.recommender.load_data()
        self.recommender.fit()
        
        # Initialize supporting components
        self.explainer = RecommendationExplainer(self.destinations, self.activities)
        self.sustainability_scorer = SustainabilityScorer(self.destinations)
        
        # Initialize chatbot
        self.chatbot = SustainableTourismChatbot(self.recommender, self.explainer)
        
        print("Recommender system initialized and trained!")
    
    def get_random_user(self) -> Dict[str, Any]:
        """Get a random user from the dataset"""
        user = self.users.sample(1).iloc[0]
        return {
            "user_id": user["user_id"],
            "age_group": user["age_group"],
            "interests": user["interests"],
            "sustainability_preference": user["sustainability_preference"],
            "travel_style": user["travel_style"]
        }
    
    def get_user_by_id(self, user_id: int) -> Dict[str, Any]:
        """Get a user by ID"""
        user = self.users[self.users["user_id"] == user_id]
        
        if len(user) == 0:
            return None
        
        user = user.iloc[0]
        
        return {
            "user_id": user["user_id"],
            "age_group": user["age_group"],
            "interests": user["interests"],
            "sustainability_preference": user["sustainability_preference"],
            "travel_style": user["travel_style"]
        }
    
    def get_recommendations(self, user_id: int, n: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for a user"""
        if self.recommender is None:
            raise ValueError("Recommender not initialized. Call initialize_recommender() first.")
        
        return self.recommender.recommend(user_id, n=n)
    
    def compare_sustainability_weighting(self, user_id: int, n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Compare recommendations with different sustainability weightings"""
        if self.recommender is None:
            raise ValueError("Recommender not initialized. Call initialize_recommender() first.")
        
        # Store original weight
        original_weight = self.recommender.sustainability_weight
        
        results = {}
        
        # Get recommendations with different weights
        for weight in [0.0, 0.3, 0.7, 1.0]:
            self.recommender.sustainability_weight = weight
            recs = self.recommender.recommend(user_id, n=n)
            results[f"weight_{weight}"] = recs
        
        # Restore original weight
        self.recommender.sustainability_weight = original_weight
        
        return results
    
    def explain_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Provide an explanation for a recommendation"""
        dest_id = recommendation["destination_id"]
        
        # Get sustainability explanation
        sustainability_explanation = self.explainer.explain_sustainability(dest_id)
        
        # Get counterfactual explanation
        counterfactual = self.explainer.generate_counterfactual_explanation(dest_id)
        
        # Find sustainable activities at this destination
        activities = self.activities[self.activities["destination_id"] == dest_id]
        sustainable_activities = activities[activities["eco_friendliness"] > 7.0]
        
        top_activities = []
        if len(sustainable_activities) > 0:
            # Sort by eco-friendliness
            sustainable_activities = sustainable_activities.sort_values("eco_friendliness", ascending=False)
            
            # Get top 3 sustainable activities
            for _, activity in sustainable_activities.head(3).iterrows():
                top_activities.append({
                    "name": activity["name"],
                    "description": activity["description"],
                    "eco_friendliness": activity["eco_friendliness"],
                    "category": activity["category"]
                })
        
        # Combine explanations
        explanation = {
            "destination": recommendation,
            "sustainability": sustainability_explanation,
            "counterfactual": counterfactual,
            "sustainable_activities": top_activities
        }
        
        return explanation
    
    def run_demo(self):
        """Run an interactive demo"""
        if self.recommender is None:
            print("Initializing recommender...")
            self.load_data()
            self.initialize_recommender()
        
        print("\n======== Sustainable Tourism Recommender System Demo ========\n")
        
        # Get a random user
        user = self.get_random_user()
        user_id = user["user_id"]
        
        print(f"Selected User (ID: {user_id}):")
        print(f"  Age Group: {user['age_group']}")
        print(f"  Interests: {', '.join(user['interests']) if isinstance(user['interests'], list) else user['interests']}")
        print(f"  Sustainability Preference: {user['sustainability_preference']} / 10")
        print(f"  Travel Style: {user['travel_style']}")
        
        print("\nGenerating recommendations...")
        recommendations = self.get_recommendations(user_id, n=5)
        
        print("\nTop 5 Recommended Destinations:")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec['name']} ({rec['country']}) - Sustainability Score: {rec['sustainability_score']:.1f}/10")
        
        # Explain first recommendation
        first_rec = recommendations[0]
        print(f"\nDetailed explanation for top recommendation: {first_rec['name']}")
        
        explanation = self.explain_recommendation(first_rec)
        
        print(f"\nSustainability Profile:")
        for metric, score in explanation["sustainability"]["metrics"].items():
            print(f"  {metric}: {score:.1f}/10")
        
        print(f"\nStrengths:")
        for strength in explanation["sustainability"]["strengths"]:
            print(f"  + {strength}")
        
        if explanation["sustainability"]["weaknesses"]:
            print(f"\nAreas for Improvement:")
            for weakness in explanation["sustainability"]["weaknesses"]:
                print(f"  - {weakness}")
        
        print(f"\nSustainable Activities at {first_rec['name']}:")
        if explanation["sustainable_activities"]:
            for activity in explanation["sustainable_activities"]:
                print(f"  • {activity['name']} - Eco-friendliness: {activity['eco_friendliness']:.1f}/10")
        else:
            print("  No highly sustainable activities found.")
        
        print("\nComparing different sustainability weightings:")
        weight_comparison = self.compare_sustainability_weighting(user_id, n=3)
        
        for weight, recs in weight_comparison.items():
            weight_val = float(weight.split("_")[1])
            avg_score = np.mean([r["sustainability_score"] for r in recs])
            print(f"  Weight {weight_val:.1f}: Avg. Sustainability = {avg_score:.2f}/10")
            for i, rec in enumerate(recs):
                print(f"    {i+1}. {rec['name']} ({rec['country']}) - {rec['sustainability_score']:.1f}/10")
        
        print("\n=== Demo Complete ===")
    
    def run_chatbot_demo(self):
        """Run an interactive chatbot demo"""
        if self.recommender is None:
            print("Initializing recommender...")
            self.load_data()
            self.initialize_recommender()
        
        print("\n======== Sustainable Tourism Chatbot Demo ========\n")
        
        # Welcome message
        print("Welcome to the Sustainable Tourism Assistant!")
        print("I can help you find eco-friendly travel destinations based on your preferences.")
        print("You can ask about destinations, sustainability features, activities, and more.")
        print("Type 'exit' or 'quit' to end our conversation.")
        print("\nChatbot: " + self.chatbot.process_message("hello"))
        
        # Chat loop
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nChatbot: Thank you for using the Sustainable Tourism Assistant. Have a great trip!")
                    break
                
                response = self.chatbot.process_message(user_input)
                print(f"\nChatbot: {response}")
            except KeyboardInterrupt:
                print("\nExiting chatbot...")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Let's continue our conversation.")


if __name__ == "__main__":
    demo = SustainableTourismDemo()
    
    # Check if chatbot mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "chatbot":
        demo.run_chatbot_demo()
    else:
        demo.run_demo()