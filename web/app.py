from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ensemble import create_default_hybrid_recommender
from src.explainability.explainations import RecommendationExplainer
from src.chatbot.chatbot_interface import SustainableTourismChatbot

app = Flask(__name__)

# Initialize recommender and chatbot
recommender = None
chatbot = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def process_message():
    global recommender, chatbot
    
    # Initialize if not already done
    if recommender is None:
        # Load data and initialize recommender
        print("Initializing recommendation system...")
        recommender = create_default_hybrid_recommender()
        recommender.load_data()
        recommender.fit()
        
        # Initialize explainer and chatbot
        explainer = RecommendationExplainer(recommender.destinations, recommender.sustainability_scorer)
        chatbot = SustainableTourismChatbot(recommender, explainer)
    
    # Get message from request
    user_message = request.json.get('message', '')
    
    # Process message with chatbot
    response = chatbot.process_message(user_message)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)