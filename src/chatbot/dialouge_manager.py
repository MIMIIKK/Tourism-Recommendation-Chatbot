from typing import Dict, List, Any, Tuple
import json
import os

class DialogueManager:
    """Manage conversation flow and state"""
    
    def __init__(self):
        self.conversation_history = []
        self.current_state = "greeting"
        self.user_profile = {}
        self.session_data = {
            "recommendations": [],
            "mentioned_destinations": [],
            "compared_destinations": [],
            "sustainability_weight": 0.5  # Default weight
        }
    
    def process_turn(self, user_message: str, system_response: str, intent: str, entities: Dict[str, Any]):
        """
        Process a conversation turn
        
        Parameters:
        - user_message: User's message
        - system_response: System's response
        - intent: Classified intent
        - entities: Extracted entities
        """
        # Add to conversation history
        self.conversation_history.append({
            "user": user_message,
            "system": system_response,
            "intent": intent,
            "entities": entities
        })
        
        # Update user profile with extracted entities
        self._update_user_profile(entities)
        
        # Update conversation state
        self._update_state(intent, entities)
    
    def _update_user_profile(self, entities: Dict[str, Any]):
        """Update user profile with extracted entities"""
        if "interests" in entities:
            if "interests" not in self.user_profile:
                self.user_profile["interests"] = []
            
            for interest in entities["interests"]:
                if interest not in self.user_profile["interests"]:
                    self.user_profile["interests"].append(interest)
        
        if "sustainability_preference" in entities:
            self.user_profile["sustainability_preference"] = entities["sustainability_preference"]
            
            # Also update session data with normalized weight
            self.session_data["sustainability_weight"] = entities["sustainability_preference"] / 10.0
        
        if "budget_level" in entities:
            self.user_profile["budget_level"] = entities["budget_level"]
        
        if "travel_style" in entities:
            self.user_profile["travel_style"] = entities["travel_style"]
        
        if "season" in entities:
            self.user_profile["preferred_season"] = entities["season"]
        
        if "duration" in entities:
            self.user_profile["trip_duration"] = entities["duration"]
    
    def _update_state(self, intent: str, entities: Dict[str, Any]):
        """Update conversation state based on intent and entities"""
        # Update state based on intent
        if intent == "greeting":
            self.current_state = "collecting_preferences"
        elif intent == "get_recommendations":
            self.current_state = "providing_recommendations"
        elif intent == "ask_about_sustainability":
            self.current_state = "explaining_sustainability"
        elif intent == "compare_destinations":
            self.current_state = "comparing_destinations"
        elif intent == "ask_about_destination":
            self.current_state = "providing_details"
        elif intent == "ask_about_activities":
            self.current_state = "suggesting_activities"
        elif intent == "farewell":
            self.current_state = "ending_conversation"
        
        # Update session data
        if intent == "get_recommendations" and "recommendations" in entities:
            self.session_data["recommendations"] = entities["recommendations"]
        
        if "destination" in entities:
            dest = entities["destination"]
            if dest not in self.session_data["mentioned_destinations"]:
                self.session_data["mentioned_destinations"].append(dest)
    
    def get_next_required_info(self) -> str:
        """Determine what information to ask for next based on current state"""
        if self.current_state == "collecting_preferences":
            # Check what information is missing
            if "interests" not in self.user_profile or not self.user_profile["interests"]:
                return "interests"
            elif "sustainability_preference" not in self.user_profile:
                return "sustainability_preference"
            elif "budget_level" not in self.user_profile:
                return "budget_level"
            else:
                return None  # All required information collected
        else:
            return None
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about the current conversation state"""
        return {
            "state": self.current_state,
            "profile": self.user_profile,
            "session_data": self.session_data,
            "history_length": len(self.conversation_history),
            "next_required_info": self.get_next_required_info()
        }
    
    def save_conversation(self, filepath: str = "conversations/"):
        """Save the conversation history to a file"""
        os.makedirs(filepath, exist_ok=True)
        
        # Generate filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filepath}/conversation_{timestamp}.json"
        
        # Save to file
        with open(filename, "w") as f:
            json.dump({
                "history": self.conversation_history,
                "user_profile": self.user_profile,
                "final_state": self.current_state
            }, f, indent=2)
        
        return filename
    
    def load_conversation(self, filepath: str) -> bool:
        """Load a conversation from a file"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            self.conversation_history = data["history"]
            self.user_profile = data["user_profile"]
            self.current_state = data["final_state"]
            
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False
    
    def reset(self):
        """Reset the conversation state"""
        self.conversation_history = []
        self.current_state = "greeting"
        self.user_profile = {}
        self.session_data = {
            "recommendations": [],
            "mentioned_destinations": [],
            "compared_destinations": [],
            "sustainability_weight": 0.5
        }