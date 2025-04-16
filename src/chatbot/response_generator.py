from typing import Dict, List, Any
import random

class ResponseGenerator:
    """Generate natural language responses for the chatbot"""
    
    def __init__(self):
        # Load response templates
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for different intents"""
        templates = {
            "greeting": [
                "Hello! I'm your sustainable tourism assistant. How can I help you find eco-friendly travel destinations today?",
                "Hi there! I'm here to help you discover sustainable travel options. What kind of trip are you planning?",
                "Welcome! I'm your guide to environmentally responsible travel. What sort of destination are you interested in?"
            ],
            "farewell": [
                "Thank you for using the Sustainable Tourism Assistant. Have a wonderful trip and safe travels!",
                "It was great helping you find sustainable travel options. Enjoy your journey!",
                "Safe travels and enjoy your sustainable adventure! Feel free to return if you need more recommendations."
            ],
            "thank_you": [
                "You're welcome! I'm happy to help with sustainable travel planning.",
                "My pleasure! Sustainable tourism benefits both travelers and destinations.",
                "Glad I could help! Enjoy your eco-friendly adventure."
            ],
            "fallback": [
                "I'm not sure I fully understood. Could you tell me more about what you're looking for in a sustainable destination?",
                "I'd like to help you better. Can you share more details about your travel preferences?",
                "I want to make sure I give you the best recommendations. Could you clarify what you're looking for?"
            ],
            "ask_for_interests": [
                "What types of activities or landscapes interest you? For example, beaches, mountains, cultural sites, or nature experiences?",
                "I'd love to know what you enjoy when traveling. Are you interested in nature, culture, adventure, or relaxation?",
                "To help find the perfect sustainable destination, what are you most interested in experiencing during your trip?"
            ],
            "ask_for_sustainability": [
                "How important is sustainability in your travel plans? This helps me prioritize destinations with stronger environmental practices.",
                "On a scale from 'somewhat important' to 'very important', how would you rate your interest in eco-friendly destinations?",
                "Would you prefer destinations with the highest possible sustainability ratings, or is that just one factor among many?"
            ],
            "ask_for_budget": [
                "What's your approximate budget range for this trip? This helps me recommend suitable sustainable options.",
                "Are you looking for budget-friendly, mid-range, or luxury sustainable accommodations?",
                "To help narrow down options, could you share your budget expectations for this journey?"
            ]
        }
        
        return templates
    
    def generate_response(self, intent: str, entities: Dict[str, Any], 
                         state_info: Dict[str, Any], data: Dict[str, Any] = None) -> str:
        """
        Generate a response based on intent, entities, and conversation state
        
        Parameters:
        - intent: Classified intent
        - entities: Extracted entities
        - state_info: Current conversation state information
        - data: Additional data for response generation (e.g., recommendations)
        
        Returns:
        - Generated response text
        """
        # Check for template responses first
        if intent in self.templates:
            response = random.choice(self.templates[intent])
            return self._fill_template(response, entities, state_info, data)
        
        # Handle specific intents that need dynamic responses
        if intent == "get_recommendations":
            return self._generate_recommendation_response(entities, state_info, data)
        elif intent == "ask_about_sustainability":
            return self._generate_sustainability_response(entities, state_info, data)
        elif intent == "compare_destinations":
            return self._generate_comparison_response(entities, state_info, data)
        elif intent == "ask_about_destination":
            return self._generate_destination_details(entities, state_info, data)
        elif intent == "ask_about_activities":
            return self._generate_activities_response(entities, state_info, data)
        elif intent == "set_preference":
            return self._generate_preference_confirmation(entities, state_info)
        elif intent == "help":
            return self._generate_help_response(state_info)
        
        # Fallback response
        return random.choice(self.templates["fallback"])
    
    def _fill_template(self, template: str, entities: Dict[str, Any], 
                      state_info: Dict[str, Any], data: Dict[str, Any] = None) -> str:
        """Fill placeholders in template with actual values"""
        # This is a simple implementation - could be expanded with more complex template filling
        response = template
        
        # Replace simple placeholders
        if "{user_name}" in response and "name" in state_info.get("profile", {}):
            response = response.replace("{user_name}", state_info["profile"]["name"])
        
        return response
    
    def _generate_recommendation_response(self, entities: Dict[str, Any], 
                                         state_info: Dict[str, Any], 
                                         data: Dict[str, Any]) -> str:
        """Generate response with destination recommendations"""
        if not data or "recommendations" not in data or not data["recommendations"]:
            return "I'd need a bit more information to make tailored recommendations. Could you tell me what kinds of activities you enjoy or what landscapes you prefer?"
        
        recommendations = data["recommendations"]
        
        response = "Based on your preferences, I recommend these sustainable destinations:\n\n"
        
        for i, rec in enumerate(recommendations[:3]):  # Limit to top 3
            response += f"{i+1}. {rec['name']} ({rec['country']}) - Sustainability Score: {rec['sustainability_score']:.1f}/10\n"
            
            # Add brief description if available
            if "description" in rec:
                response += f"   {rec['description']}\n"
            elif "landscape" in rec:
                response += f"   Known for its {rec['landscape'].lower()} landscapes.\n"
            
            response += "\n"
        
        response += "Would you like to know more about any of these destinations or see how they compare?"
        
        return response
    
    def _generate_sustainability_response(self, entities: Dict[str, Any], 
                                         state_info: Dict[str, Any], 
                                         data: Dict[str, Any]) -> str:
        """Generate response explaining sustainability aspects"""
        # Check if a specific destination was mentioned
        destination = None
        if "destination" in entities:
            destination = entities["destination"]
        elif state_info["session_data"]["mentioned_destinations"]:
            # Use the most recently mentioned destination
            destination = state_info["session_data"]["mentioned_destinations"][-1]
        
        if not destination:
            # General sustainability explanation
            return ("Sustainable tourism minimizes negative environmental impacts while supporting local communities. "
                   "Key aspects include reducing carbon emissions, conserving water and energy, minimizing waste, "
                   "protecting biodiversity, and ensuring economic benefits reach local people. "
                   "Would you like me to recommend destinations with strong sustainability practices?")
        
        # Destination-specific explanation
        response = f"About the sustainability of {destination['name']}:\n\n"
        
        # Use detailed data if available
        if data and "sustainability_info" in data:
            info = data["sustainability_info"]
            
            if "highlights" in info:
                response += "🟢 Sustainability highlights:\n"
                for highlight in info["highlights"]:
                    response += f"• {highlight}\n"
                response += "\n"
            
            if "metrics" in info:
                response += "📊 Sustainability metrics:\n"
                for metric, value in info["metrics"].items():
                    formatted_metric = metric.replace("_", " ").title()
                    response += f"• {formatted_metric}: {value:.1f}/10\n"
                response += "\n"
            
            if "initiatives" in info:
                response += "🌱 Key initiatives:\n"
                for initiative in info["initiatives"]:
                    response += f"• {initiative}\n"
                response += "\n"
        else:
            # Generic response
            response += "This destination focuses on several sustainability aspects:\n"
            response += "• Reducing carbon footprint through renewable energy and efficient transportation\n"
            response += "• Water conservation initiatives and waste reduction programs\n"
            response += "• Supporting local communities through fair employment and cultural preservation\n"
            response += "• Protecting biodiversity and natural habitats\n\n"
        
        response += "Would you like to know about eco-friendly activities at this destination?"
        
        return response
    
    def _generate_comparison_response(self, entities: Dict[str, Any], 
                                     state_info: Dict[str, Any], 
                                     data: Dict[str, Any]) -> str:
        """Generate response comparing destinations"""
        if not data or "destinations_to_compare" not in data or len(data["destinations_to_compare"]) < 2:
            return "I'd need at least two destinations to compare. Would you like me to recommend some options first?"
        
        dest1, dest2 = data["destinations_to_compare"][:2]
        
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
        
        # Add other relevant comparisons if data available
        if "detailed_comparison" in data:
            for aspect, comparison in data["detailed_comparison"].items():
                response += f"{aspect}: {comparison}\n"
            response += "\n"
        
        response += "Would you like to know more about either of these destinations?"
        
        return response
    
    def _generate_destination_details(self, entities: Dict[str, Any], 
                                     state_info: Dict[str, Any], 
                                     data: Dict[str, Any]) -> str:
        """Generate detailed information about a destination"""
        if not data or "destination" not in data:
            return "Which destination would you like to know more about? You can refer to one of my recommendations."
        
        destination = data["destination"]
        
        response = f"About {destination['name']}:\n\n"
        
        # Add destination details
        if "country" in destination:
            response += f"Location: {destination['country']}\n"
        
        if "landscape" in destination:
            response += f"Landscape: {destination['landscape']}\n"
        
        # Add sustainability score
        if "sustainability_score" in destination:
            response += f"Sustainability Score: {destination['sustainability_score']:.1f}/10\n\n"
        
        # Add detailed description if available
        if "full_description" in data:
            response += data["full_description"] + "\n\n"
        else:
            # Generic description
            response += f"{destination['name']} offers a wonderful balance of natural beauty and sustainable tourism practices. "
            response += "The local community is involved in tourism management, ensuring that your visit contributes positively to the area. "
            
            # Add landscape-specific description
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
    
    def _generate_activities_response(self, entities: Dict[str, Any], 
                                     state_info: Dict[str, Any], 
                                     data: Dict[str, Any]) -> str:
        """Generate information about sustainable activities at a destination"""
        if not data or "destination" not in data:
            return "Which destination's activities would you like to know about? You can refer to one of my recommendations."
        
        destination = data["destination"]
        
        response = f"Sustainable activities in {destination['name']}:\n\n"
        
        # Use actual activities if available
        if "activities" in data and data["activities"]:
            activities = data["activities"]
            
            # Group by category if available
            categorized = {}
            for activity in activities:
                if "category" in activity:
                    cat = activity["category"]
                    if cat not in categorized:
                        categorized[cat] = []
                    categorized[cat].append(activity)
            
            if categorized:
                for category, acts in categorized.items():
                    response += f"📍 {category}:\n"
                    for act in acts[:3]:  # Limit to 3 per category
                        response += f"• {act['name']}"
                        if "eco_friendliness" in act:
                            response += f" - Eco-friendly rating: {act['eco_friendliness']:.1f}/10"
                        response += "\n"
                    response += "\n"
            else:
                # List without categories
                for activity in activities[:5]:  # Limit to top 5
                    response += f"• {activity['name']}"
                    if "eco_friendliness" in activity:
                        response += f" - Eco-friendly rating: {activity['eco_friendliness']:.1f}/10"
                    response += "\n"
        else:
            # Generic activities based on landscape
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
    
    def _generate_preference_confirmation(self, entities: Dict[str, Any], 
                                         state_info: Dict[str, Any]) -> str:
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
        
        if "travel_style" in entities:
            response += f"You prefer {entities['travel_style']} travel. "
        
        response += "\nWould you like me to recommend some destinations based on these preferences?"
        
        return response
    
    def _generate_help_response(self, state_info: Dict[str, Any]) -> str:
        """Generate help information based on current state"""
        response = "I'm your sustainable tourism assistant. Here's how I can help you:\n\n"
        response += "• Find eco-friendly travel destinations based on your preferences\n"
        response += "• Explain sustainability features of destinations\n"
        response += "• Compare different destinations on sustainability factors\n"
        response += "• Suggest sustainable activities at your chosen destinations\n"
        response += "• Provide detailed information about destinations\n\n"
        
        current_state = state_info.get("state", "greeting")
        
        if current_state == "greeting" or current_state == "collecting_preferences":
            response += "To get started, tell me what kind of trip you're looking for and any preferences you have for your travel destination."
        elif current_state == "providing_recommendations":
            response += "I've provided some recommendations. You can ask for more details about any destination, compare options, or ask about sustainability features."
        elif current_state in ["explaining_sustainability", "providing_details", "suggesting_activities"]:
            response += "You can ask me to recommend more destinations at any time, or tell me more about your preferences to get better recommendations."
        
        return response
    
    def get_next_question(self, required_info: str) -> str:
        """Get the next question to ask based on required information"""
        if required_info == "interests":
            return random.choice(self.templates["ask_for_interests"])
        elif required_info == "sustainability_preference":
            return random.choice(self.templates["ask_for_sustainability"])
        elif required_info == "budget_level":
            return random.choice(self.templates["ask_for_budget"])
        else:
            return "Can you tell me more about what you're looking for in a destination?"