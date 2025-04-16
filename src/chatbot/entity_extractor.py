import re
import spacy
from typing import Dict, List, Any

class EntityExtractor:
    """Extract entities from user messages"""
    
    def __init__(self, use_spacy: bool = False):
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Spacy model not found. Using rule-based extraction instead.")
                self.use_spacy = False
        
        # Initialize keyword dictionaries
        self.initialize_keywords()
    
    def initialize_keywords(self):
        """Initialize keyword dictionaries for entity extraction"""
        # Interest keywords
        self.interest_keywords = {
            "beach": ["beach", "ocean", "sea", "sand", "coastal", "swimming", "sunbathing"],
            "mountain": ["mountain", "hiking", "climbing", "altitude", "trek", "hill", "peak"],
            "culture": ["culture", "history", "museum", "art", "heritage", "architecture", "historical"],
            "nature": ["nature", "wildlife", "forest", "national park", "animals", "outdoors", "scenic"],
            "adventure": ["adventure", "exciting", "thrill", "extreme", "adrenaline", "sports", "active"],
            "relaxation": ["relax", "peaceful", "calm", "tranquil", "quiet", "spa", "rest", "unwind"],
            "food": ["food", "cuisine", "gastronomy", "culinary", "dining", "restaurant", "eat", "taste"],
            "urban": ["city", "urban", "metropolitan", "shopping", "nightlife", "modern"]
        }
        
        # Sustainability preference phrases
        self.sustainability_phrases = {
            "very high": ["extremely sustainable", "most eco", "greenest", "most sustainable", "very eco",
                         "highest sustainability", "completely sustainable", "fully eco"],
            "high": ["very sustainable", "eco-friendly", "green", "environmentally friendly", "sustainable",
                    "environmentally conscious", "eco"],
            "medium": ["somewhat sustainable", "reasonably eco", "fairly green", "moderately sustainable",
                      "some eco options", "partially sustainable"],
            "low": ["not too concerned about sustainability", "sustainability isn't priority", 
                   "not very eco", "less eco", "not focused on sustainability"]
        }
        
        # Budget level phrases
        self.budget_phrases = {
            "luxury": ["luxury", "high-end", "five star", "premium", "expensive", "upscale", "top-tier"],
            "mid-range": ["mid range", "moderate", "average", "standard", "reasonable", "middle", "not too expensive"],
            "budget": ["budget", "cheap", "affordable", "inexpensive", "low cost", "economical", "thrifty"]
        }
        
        # Travel styles
        self.travel_styles = {
            "solo": ["solo", "alone", "by myself", "independent"],
            "couple": ["couple", "romantic", "honeymoon", "anniversary", "with partner"],
            "family": ["family", "with kids", "children", "family-friendly"],
            "friends": ["friends", "group", "with buddies", "with pals"],
            "business": ["business", "work", "conference", "meeting"]
        }
        
        # Season preferences
        self.seasons = {
            "summer": ["summer", "hot", "warm", "july", "august", "june"],
            "winter": ["winter", "snow", "cold", "december", "january", "february"],
            "spring": ["spring", "blossom", "flowers", "march", "april", "may"],
            "fall": ["fall", "autumn", "foliage", "september", "october", "november"]
        }
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities from user message
        
        Parameters:
        - message: User message text
        
        Returns:
        - Dictionary of extracted entities
        """
        if self.use_spacy and self.nlp:
            return self._extract_with_spacy(message)
        else:
            return self._extract_with_rules(message)
    
    def _extract_with_rules(self, message: str) -> Dict[str, Any]:
        """Extract entities using rule-based approach"""
        entities = {}
        message = message.lower()
        
        # Save original message
        entities["raw_text"] = message
        
        # Extract interests
        interests = []
        for interest, keywords in self.interest_keywords.items():
            if any(re.search(rf'\b{keyword}\b', message) for keyword in keywords):
                interests.append(interest)
        
        if interests:
            entities["interests"] = interests
        
        # Extract sustainability preference
        for level, phrases in self.sustainability_phrases.items():
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
        for level, phrases in self.budget_phrases.items():
            if any(re.search(rf'\b{phrase}\b', message) for phrase in phrases):
                if level == "luxury":
                    entities["budget_level"] = 5
                elif level == "mid-range":
                    entities["budget_level"] = 3
                elif level == "budget":
                    entities["budget_level"] = 1
        
        # Extract travel style
        for style, phrases in self.travel_styles.items():
            if any(re.search(rf'\b{phrase}\b', message) for phrase in phrases):
                entities["travel_style"] = style
        
        # Extract season preference
        for season, phrases in self.seasons.items():
            if any(re.search(rf'\b{phrase}\b', message) for phrase in phrases):
                entities["season"] = season
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s*(day|days|week|weeks|month|months)', message)
        if duration_match:
            value = int(duration_match.group(1))
            unit = duration_match.group(2)
            
            if unit in ["week", "weeks"]:
                value *= 7
            elif unit in ["month", "months"]:
                value *= 30
            
            entities["duration"] = value
        
        return entities
    
    def _extract_with_spacy(self, message: str) -> Dict[str, Any]:
        """Extract entities using spaCy NLP"""
        entities = self._extract_with_rules(message)  # Get rule-based entities as a base
        
        # Parse with spaCy
        doc = self.nlp(message)
        
        # Extract locations
        locations = []
        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                locations.append(ent.text)
        
        if locations:
            entities["locations"] = locations
        
        # Extract dates
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text)
        
        if dates:
            entities["dates"] = dates
        
        return entities