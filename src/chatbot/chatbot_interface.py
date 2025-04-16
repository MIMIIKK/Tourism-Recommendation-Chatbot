import re
import numpy as np
from typing import Dict, List, Any, Tuple
import os

class SustainableTourismChatbot:
    """Chatbot interface for the Sustainable Tourism Recommender System"""
    
    def __init__(self, recommender=None, explainer=None):
        self.recommender = recommender
        self.explainer = explainer
        self.user_profile = {
            "sustainability_preference": 7.0,  # Default medium-high preference
            "interests": [],
            "travel_style": None,
            "budget_level": 3,  # Default mid-range budget (1-5)
            "previous_destinations": []
        }
        self.context = {
            "current_recommendations": [],
            "last_intent": None,
            "mentioned_destinations": [],
            "comparison_destinations": []
        }
        self.user_id = 1  # Default user ID for recommendations
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and return a response
        
        Parameters:
        - message: User's message text
        
        Returns:
        - Chatbot response
        """
        # Check for conversation reset command
        if message.lower() in ["restart", "reset", "start over"]:
            self._reset_context()
            return "I've reset our conversation. What kind of trip are you interested in?"
        
        # Classify intent
        intent = self._classify_intent(message)
        self.context["last_intent"] = intent
        
        # Extract entities
        entities = self._extract_entities(message)
        
        # Update user profile with any extracted preferences
        self._update_user_profile(entities)
        
        # Generate response based on intent
        if intent == "greeting":
            return self._greeting_response()
        elif intent == "get_recommendations":
            return self._recommendation_response(entities)
        elif intent == "ask_about_sustainability":
            return self._sustainability_explanation_response(entities)
        elif intent == "compare_destinations":
            return self._comparison_response(entities)
        elif intent == "set_preference":
            return self._preference_confirmation_response(entities)
        elif intent == "ask_about_destination":
            return self._destination_details_response(entities)
        elif intent == "ask_about_activities":
            return self._activities_response(entities)
        elif intent == "farewell":
            return "Thank you for using the Sustainable Tourism Assistant. Have a wonderful trip and safe travels!"
        else:
            return "I'm here to help you find sustainable travel destinations. Could you tell me more about what kind of trip you're looking for? For example, are you interested in beaches, mountains, or cultural experiences?"
    
    def _classify_intent(self, message: str) -> str:
        """Simple rule-based intent classification"""
        message = message.lower()
        
        if re.search(r'\bhello\b|\bhi\b|\bhey\b|\bgreetings\b', message):
            return "greeting"
        elif re.search(r'\brecommend|\bsuggest|\bwhere should|\btravel to', message):
            return "get_recommendations"
        elif re.search(r'\bsustainable|\beco-friendly|\bgreen|\benvironment', message):
            return "ask_about_sustainability"
        elif re.search(r'\bcompare|\bdifference|\bversus|\bvs\b', message):
            return "compare_destinations"
        elif re.search(r'\bprefer|\blike|\bwant|\binterested|\bbudget', message):
            return "set_preference"
        elif re.search(r'\btell me about|\bmore about|\bdetails|\binformation|\bwhat is', message):
            return "ask_about_destination"
        elif re.search(r'\bactivities|\bthings to do|\bexperiences|\btours|\battractions', message):
            return "ask_about_activities"
        elif re.search(r'\bgoodbye|\bbye|\bthanks|\bthank you|\bsee you', message):
            return "farewell"
        else:
            return "general_query"
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract relevant entities from the message"""
        entities = {}
        message = message.lower()
        
        # Extract interests
        interests = []
        interest_keywords = {
            "beach": ["beach", "ocean", "sea", "sand", "coastal"],
            "mountain": ["mountain", "hiking", "climbing", "altitude", "trek"],
            "culture": ["culture", "history", "museum", "art", "heritage"],
            "nature": ["nature", "wildlife", "forest", "national park", "animals"],
            "adventure": ["adventure", "exciting", "thrill", "extreme", "adrenaline"],
            "relaxation": ["relax", "peaceful", "calm", "tranquil", "quiet"]
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in message for keyword in keywords):
                interests.append(interest)
        
        if interests:
            entities["interests"] = interests
        
        # Extract sustainability preference
        sustainability_phrases = {
            "very high": ["extremely sustainable", "most eco", "greenest", "most sustainable", "very eco"],
            "high": ["very sustainable", "eco-friendly", "green", "environmentally friendly", "sustainable"],
            "medium": ["somewhat sustainable", "reasonably eco", "fairly green"],
            "low": ["not too concerned about sustainability", "sustainability isn't priority"]
        }
        
        for level, phrases in sustainability_phrases.items():
            if any(phrase in message for phrase in phrases):
                if level == "very high":
                    entities["sustainability_preference"] = 9.5
                elif level == "high":
                    entities["sustainability_preference"] = 8.0
                elif level == "medium":
                    entities["sustainability_preference"] = 6.0
                elif level == "low":
                    entities["sustainability_preference"] = 4.0
        
        # Extract budget level
        budget_phrases = {
            "luxury": ["luxury", "high-end", "five star", "premium", "expensive"],
            "mid-range": ["mid range", "moderate", "average", "standard", "reasonable"],
            "budget": ["budget", "cheap", "affordable", "inexpensive", "low cost"]
        }
        
        for level, phrases in budget_phrases.items():
            if any(phrase in message for phrase in phrases):
                if level == "luxury":
                    entities["budget_level"] = 5
                elif level == "mid-range":
                    entities["budget_level"] = 3
                elif level == "budget":
                    entities["budget_level"] = 1
        
        # Extract destination references from current recommendations
        if self.context["current_recommendations"]:
            for rec in self.context["current_recommendations"]:
                dest_name = rec["name"].lower()
                if dest_name in message:
                    entities["destination"] = rec
                    self.context["mentioned_destinations"].append(rec)
                    break
        
        # Extract numbers for referenced recommendations (e.g., "tell me about number 2")
        number_match = re.search(r'\bnumber (\d+)\b|\boption (\d+)\b|\b(\d+)\b', message)
        if number_match:
            # Get the number from whichever group matched
            num = int(next(group for group in number_match.groups() if group is not None))
            if 1 <= num <= len(self.context["current_recommendations"]):
                entities["destination"] = self.context["current_recommendations"][num-1]
                self.context["mentioned_destinations"].append(self.context["current_recommendations"][num-1])
        
        return entities
    
    def _update_user_profile(self, entities: Dict[str, Any]):
        """Update user profile with extracted entities"""
        if "interests" in entities:
            for interest in entities["interests"]:
                if interest not in self.user_profile["interests"]:
                    self.user_profile["interests"].append(interest)
        
        if "sustainability_preference" in entities:
            self.user_profile["sustainability_preference"] = entities["sustainability_preference"]
        
        if "budget_level" in entities:
            self.user_profile["budget_level"] = entities["budget_level"]
        
        if "travel_style" in entities:
            self.user_profile["travel_style"] = entities["travel_style"]
    
    def _reset_context(self):
        """Reset the conversation context"""
        self.context = {
            "current_recommendations": [],
            "last_intent": None,
            "mentioned_destinations": [],
            "comparison_destinations": []
        }
    
    def _greeting_response(self) -> str:
        """Generate a greeting response"""
        return ("Hello! I'm your sustainable tourism assistant. I can help you find eco-friendly travel destinations "
                "based on your preferences. What kind of trip are you looking for? Are you interested in beaches, "
                "mountains, cultural experiences, or something else?")
    
    def _recommendation_response(self, entities: Dict[str, Any]) -> str:
        """Generate destination recommendations"""
        if not self.recommender:
            return ("I'd love to give you recommendations, but my recommendation engine isn't connected yet. "
                    "Tell me more about your preferences, and I'll note them down for future recommendations.")
        
        # Apply sustainability preference to recommender
        sustainability_weight = self.user_profile["sustainability_preference"] / 10.0
        self.recommender.sustainability_weight = sustainability_weight
        
        # Get recommendations
        recommendations = self.recommender.recommend(self.user_id, n=3)
        self.context["current_recommendations"] = recommendations
        
        # Format response
        response = "Based on your preferences, I recommend these sustainable destinations:\n\n"
        
        for i, rec in enumerate(recommendations):
            response += f"{i+1}. {rec['name']} ({rec['country']}) - Sustainability Score: {rec['sustainability_score']:.1f}/10\n"
            
            # Add a brief description based on destination attributes
            if "landscape" in rec:
                response += f"   Known for its {rec['landscape'].lower()} landscapes"
                
            response += ".\n\n"
        
        response += "Would you like to know more about any of these destinations or compare them? You can say 'tell me more about option 1' or 'compare options 1 and 2'."
        
        return response
    
    def _sustainability_explanation_response(self, entities: Dict[str, Any]) -> str:
        """Generate explanation about sustainability aspects"""
        # Check if a specific destination was mentioned
        destination = None
        if "destination" in entities:
            destination = entities["destination"]
        elif self.context["mentioned_destinations"]:
            # Use the most recently mentioned destination
            destination = self.context["mentioned_destinations"][-1]
        elif self.context["current_recommendations"]:
            # Use the first recommendation as default
            destination = self.context["current_recommendations"][0]
        
        if not destination:
            return ("Sustainable tourism minimizes negative environmental impacts while supporting local communities. "
                    "I can tell you about the sustainability features of specific destinations if you mention one.")
        
        # If explainer is available, use it to get detailed sustainability information
        if self.explainer and "destination_id" in destination:
            try:
                sustainability_info = self.explainer.explain_sustainability(destination["destination_id"])
                
                response = f"About the sustainability of {destination['name']}:\n\n"
                
                # Format strengths
                if "strengths" in sustainability_info and sustainability_info["strengths"]:
                    response += "🟢 Sustainability strengths:\n"
                    for strength in sustainability_info["strengths"]:
                        response += f"• {strength}\n"
                    response += "\n"
                
                # Format metrics
                if "metrics" in sustainability_info:
                    response += "📊 Sustainability metrics:\n"
                    for metric, value in sustainability_info["metrics"].items():
                        formatted_metric = metric.replace("_", " ").title()
                        response += f"• {formatted_metric}: {value:.1f}/10\n"
                    response += "\n"
                
                # Format certifications
                if "certifications" in sustainability_info and sustainability_info["certifications"]:
                    response += "🏆 Certifications:\n"
                    for cert in sustainability_info["certifications"]:
                        if cert != "None":
                            response += f"• {cert}\n"
                    response += "\n"
                
                response += "Would you like to know about eco-friendly activities at this destination?"
                
                return response
            
            except Exception as e:
                # Fall back to generic response if there's an error
                pass
        
        # Generic response if explainer not available or failed
        response = f"About the sustainability of {destination['name']}:\n\n"
        response += "This destination focuses on several sustainability aspects:\n"
        response += "• Reducing carbon footprint through renewable energy and efficient transportation\n"
        response += "• Water conservation initiatives and waste reduction programs\n"
        response += "• Supporting local communities through fair employment and cultural preservation\n"
        response += "• Protecting biodiversity and natural habitats\n\n"
        response += "Would you like specific recommendations for eco-friendly activities at this destination?"
        
        return response
    
    def _comparison_response(self, entities: Dict[str, Any]) -> str:
        """Generate comparison between destinations"""
        if len(self.context["current_recommendations"]) < 2:
            return "I need to provide you with at least two destination options before I can compare them. Would you like some recommendations first?"
        
        # Compare the first two recommendations
        dest1 = self.context["current_recommendations"][0]
        dest2 = self.context["current_recommendations"][1]
        
        # Check if entities contain specific destinations to compare
        dest_indices = []
        number_matches = re.findall(r'\b(\d+)\b', entities.get("raw_text", ""))
        
        if number_matches:
            for num in number_matches:
                idx = int(num) - 1
                if 0 <= idx < len(self.context["current_recommendations"]):
                    dest_indices.append(idx)
        
        # Use specified destinations if available
        if len(dest_indices) >= 2:
            dest1 = self.context["current_recommendations"][dest_indices[0]]
            dest2 = self.context["current_recommendations"][dest_indices[1]]
        
        response = f"Comparing {dest1['name']} and {dest2['name']}:\n\n"
        
        # Sustainability comparison
        response += f"Sustainability: {dest1['name']}: {dest1['sustainability_score']:.1f}/10 vs. {dest2['name']}: {dest2['sustainability_score']:.1f}/10\n"
        
        if dest1['sustainability_score'] > dest2['sustainability_score']:
            diff = dest1['sustainability_score'] - dest2['sustainability_score']
            response += f"{dest1['name']} is more sustainable with a {diff:.1f} point higher score.\n\n"
        elif dest2['sustainability_score'] > dest1['sustainability_score']:
            diff = dest2['sustainability_score'] - dest1['sustainability_score']
            response += f"{dest2['name']} is more sustainable with a {diff:.1f} point higher score.\n\n"
        else:
            response += f"Both destinations have similar sustainability scores.\n\n"
        
        # Add landscape comparison if available
        if "landscape" in dest1 and "landscape" in dest2:
            response += f"Landscape: {dest1['name']} offers {dest1['landscape']} landscapes, while {dest2['name']} features {dest2['landscape']} scenery.\n\n"
        
        # Add country comparison
        if "country" in dest1 and "country" in dest2:
            if dest1['country'] == dest2['country']:
                response += f"Both destinations are located in {dest1['country']}.\n\n"
            else:
                response += f"Location: {dest1['name']} is in {dest1['country']}, while {dest2['name']} is in {dest2['country']}.\n\n"
        
        response += "Would you like to know more about either of these destinations?"
        
        return response
    
    def _preference_confirmation_response(self, entities: Dict[str, Any]) -> str:
        """Confirm updated preferences"""
        response = "I've updated your preferences. "
        
        if "interests" in entities:
            interests_str = ", ".join(entities["interests"])
            response += f"You're interested in {interests_str}. "
        
        if "sustainability_preference" in entities:
            level = "very high" if entities["sustainability_preference"] > 9 else \
                   "high" if entities["sustainability_preference"] > 7 else \
                   "medium" if entities["sustainability_preference"] > 5 else "low"
            response += f"I've noted your {level} interest in sustainability. "
        
        if "budget_level" in entities:
            budget = "luxury" if entities["budget_level"] == 5 else \
                    "mid-range" if entities["budget_level"] == 3 else "budget"
            response += f"You're looking for {budget} options. "
        
        response += "\nWould you like me to recommend some destinations based on these preferences?"
        
        return response
    
    def _destination_details_response(self, entities: Dict[str, Any]) -> str:
        """Generate detailed information about a destination"""
        # Check if a specific destination was mentioned
        destination = None
        if "destination" in entities:
            destination = entities["destination"]
        elif self.context["mentioned_destinations"]:
            # Use the most recently mentioned destination
            destination = self.context["mentioned_destinations"][-1]
        
        if not destination:
            return "Which destination would you like to know more about? You can refer to one of the recommendations I provided."
        
        response = f"About {destination['name']}:\n\n"
        
        # Add destination details
        if "country" in destination:
            response += f"Location: {destination['country']}\n"
        
        if "landscape" in destination:
            response += f"Landscape: {destination['landscape']}\n"
        
        # Add sustainability score
        if "sustainability_score" in destination:
            response += f"Sustainability Score: {destination['sustainability_score']:.1f}/10\n\n"
        
        # Add generic information
        response += f"{destination['name']} offers a wonderful balance of natural beauty and sustainable tourism practices. "
        response += "The local community is involved in tourism management, ensuring that your visit contributes positively to the area. "
        
        if "landscape" in destination:
            landscape = destination["landscape"].lower()
            if landscape == "beach" or landscape == "coastal":
                response += "The pristine beaches are protected through conservation efforts, and many accommodations use renewable energy sources."
            elif landscape == "mountain":
                response += "The mountain ecosystem is carefully preserved, with regulated hiking trails and wildlife protection programs."
            elif landscape == "forest":
                response += "The forest biodiversity is protected through conservation initiatives, and eco-tours are designed to minimize environmental impact."
            elif landscape == "city":
                response += "The city has invested in green spaces, public transportation, and energy-efficient buildings to reduce its environmental footprint."
        
        response += "\n\nWould you like to know about sustainable activities in this destination or its specific sustainability initiatives?"
        
        return response
    
    def _activities_response(self, entities: Dict[str, Any]) -> str:
        """Generate information about sustainable activities at a destination"""
        # Check if a specific destination was mentioned
        destination = None
        if "destination" in entities:
            destination = entities["destination"]
        elif self.context["mentioned_destinations"]:
            # Use the most recently mentioned destination
            destination = self.context["mentioned_destinations"][-1]
        elif self.context["current_recommendations"]:
            # Use the first recommendation as default
            destination = self.context["current_recommendations"][0]
        
        if not destination:
            return "Which destination's activities would you like to know about? You can refer to one of the recommendations I provided."
        
        response = f"Sustainable activities in {destination['name']}:\n\n"
        
        # Generic sustainable activities based on landscape
        if "landscape" in destination:
            landscape = destination["landscape"].lower()
            
            if landscape == "beach" or landscape == "coastal":
                response += "🏄‍♂️ Eco-friendly water activities:\n"
                response += "• Guided snorkeling tours with marine conservation focus\n"
                response += "• Beach clean-up volunteer opportunities\n"
                response += "• Sustainable sailing excursions\n\n"
                
                response += "🌱 Nature experiences:\n"
                response += "• Coastal ecosystem educational walks\n"
                response += "• Sustainable fishing trips with local guides\n"
                
            elif landscape == "mountain":
                response += "🥾 Responsible hiking experiences:\n"
                response += "• Guided nature hikes on maintained trails\n"
                response += "• Wildlife observation with conservation experts\n"
                response += "• Mountain biking on designated routes\n\n"
                
                response += "🌲 Environmental education:\n"
                response += "• Alpine ecosystem workshops\n"
                response += "• Sustainable foraging with local experts\n"
                
            elif landscape == "forest":
                response += "🌳 Forest immersion experiences:\n"
                response += "• Guided forest bathing sessions\n"
                response += "• Birdwatching tours with conservation focus\n"
                response += "• Sustainable foraging workshops\n\n"
                
                response += "🦋 Environmental education:\n"
                response += "• Biodiversity workshops\n"
                response += "• Tree planting initiatives\n"
                
            elif landscape == "city":
                response += "🚲 Eco-friendly urban exploration:\n"
                response += "• Bike tours of sustainable city initiatives\n"
                response += "• Green architecture walking tours\n"
                response += "• Public transportation day passes\n\n"
                
                response += "🥗 Sustainable gastronomy:\n"
                response += "• Farm-to-table restaurant experiences\n"
                response += "• Local food market tours\n"
        else:
            # Generic activities if landscape not available
            response += "• Guided nature experiences with local experts\n"
            response += "• Cultural immersion with community benefits\n"
            response += "• Eco-certified tours and activities\n"
            response += "• Farm-to-table dining experiences\n"
            response += "• Low-impact transportation options\n"
        
        response += "\n\nAll these activities are designed to minimize environmental impact while maximizing benefits to local communities. Would you like more specific information about any of these activities?"
        
        return response