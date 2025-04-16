import re
from typing import Dict, List, Any, Tuple

class IntentClassifier:
    """Classify user intent from messages"""
    
    def __init__(self):
        # Initialize intent patterns
        self.intent_patterns = {
            "greeting": [
                r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgreetings\b', r'^hi$', r'^hello$', r'^hey$',
                r'\bgood\s(morning|afternoon|evening)\b', r'\bhowdy\b'
            ],
            "farewell": [
                r'\bgoodbye\b', r'\bbye\b', r'\bsee you\b', r'^bye$', r'^goodbye$',
                r'\bthanks for your help\b', r'\bthanks,?\s*bye\b', r'\bthank you,?\s*bye\b'
            ],
            "thank_you": [
                r'\bthanks?\b', r'\bthank\s+you\b', r'\bappreciate\b', r'\bgrateful\b'
            ],
            "get_recommendations": [
                r'\brecommend\b', r'\bsuggest\b', r'\bwhere should\b', r'\btravel to\b',
                r'\bdestination\b', r'\bplace to visit\b', r'\bwhere to go\b', r'\bwhere can i\b',
                r'\bwhat(?:\s+\w+){0,3}\s+recommend\b', r'\bshow me\b'
            ],
            "ask_about_sustainability": [
                r'\bsustainable\b', r'\beco-friendly\b', r'\bgreen\b', r'\benvironment',
                r'\benvironmental impact\b', r'\bcarbon footprint\b', r'\beco\b',
                r'\bhow sustainable\b', r'\bsustainability\b'
            ],
            "compare_destinations": [
                r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b',
                r'\bwhich is\s+(?:\w+\s+){0,3}better\b', r'\bwhich one\b', 
                r'\bwhich would\b', r'\bwhich should\b', r'\bbetter choice\b'
            ],
            "set_preference": [
                r'\bprefer\b', r'\blike\b', r'\bwant\b', r'\binterested\b', r'\bbudget\b',
                r'\bfavor\b', r'\bfancy\b', r'\bcare\s+about\b', r'\bvalue\b', r'\bimportant to me\b',
                r'\blooking for\b', r'\bseeking\b'
            ],
            "ask_about_destination": [
                r'\btell me about\b', r'\bmore about\b', r'\bdetails\b', r'\binformation\b', 
                r'\bwhat is\b', r'\bfeatures\b', r'\bdescribe\b', r'\blearning about\b',
                r'\btell me more\b'
            ],
            "ask_about_activities": [
                r'\bactivities\b', r'\bthings to do\b', r'\bexperiences\b', r'\btours\b', 
                r'\battractions\b', r'\bsightseeing\b', r'\badventures\b', r'\bfun\b',
                r'\bwhat can i do\b', r'\bwhat to do\b'
            ],
            "help": [
                r'\bhelp\b', r'\bassist\b', r'\bconfused\b', r'\bdon\'t understand\b',
                r'\bhow does this work\b', r'\bwhat can you do\b', r'\bwhat are you\b',
                r'\bhow can you\b', r'\bwhat is this\b'
            ],
            "reset": [
                r'\breset\b', r'\brestart\b', r'\bstart over\b', r'\bnew search\b',
                r'\bbest time to go\b'
            ]
        }
    
    def classify_intent(self, message: str) -> str:
        """
        Classify the intent of a user message
        
        Parameters:
        - message: User message text
        
        Returns:
        - Intent string
        """
        message = message.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    return intent
        
        # Default intent if no pattern matched
        return "general_query"
    
    def classify_with_confidence(self, message: str) -> List[Tuple[str, float]]:
        """
        Classify intent with confidence scores
        
        Parameters:
        - message: User message text
        
        Returns:
        - List of (intent, confidence) tuples, sorted by confidence
        """
        message = message.lower()
        results = []
        
        # Calculate matches for each intent
        for intent, patterns in self.intent_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, message):
                    matches += 1
            
            if matches > 0:
                confidence = min(1.0, matches / len(patterns) + 0.3)  # Base confidence boost
                results.append((intent, confidence))
        
        if not results:
            results.append(("general_query", 0.3))
        
        # Sort by confidence
        return sorted(results, key=lambda x: x[1], reverse=True)