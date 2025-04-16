import os
import sys
import argparse
import pandas as pd
import numpy as np
from time import time

from src.data.synthetic_data_generator import SyntheticDataGenerator
from src.data.data_processor import DataProcessor
from src.models.base_models import PopularityRecommender, ContentBasedRecommender, CollaborativeFilteringRecommender
from src.models.neural_cf import NeuralCollaborativeFiltering
from src.models.ensemble import create_default_hybrid_recommender
from src.evaluation.metrics import compare_recommenders
from demo.app import SustainableTourismDemo

def generate_data(args):
    """Generate synthetic data"""
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(
        num_destinations=args.destinations,
        num_activities=args.activities,
        num_users=args.users
    )
    generator.save_data()
    print("Data generation complete!")

def process_data(args):
    """Process raw data"""
    print("Processing data...")
    processor = DataProcessor()
    processor.process_all()
    print("Data processing complete!")

def train_models(args):
    """Train recommendation models"""
    print("Training recommendation models...")
    
    # Initialize models
    models = [
        PopularityRecommender(),
        ContentBasedRecommender(),
        CollaborativeFilteringRecommender(method="user"),
        CollaborativeFilteringRecommender(method="item")
    ]
    
    # Add neural CF if specified
    if args.neural:
        models.append(NeuralCollaborativeFiltering(embedding_dim=32, hidden_layers=[64, 32]))
    
    # Add hybrid recommender
    hybrid = create_default_hybrid_recommender()
    models.append(hybrid)
    
    # Load data and train each model
    for model in models:
        print(f"Training {model.name}...")
        model.load_data()
        model.fit()
    
    print("All models trained!")
    
    # Save neural model if trained
    if args.neural:
        models[-2].save_model()
    
    return models

def evaluate(args):
    """Evaluate recommendation models"""
    print("Evaluating models...")
    
    # Train models
    models = train_models(args)
    
    # Load user data
    users = pd.read_pickle("data/processed/users.pkl")
    
    # Select random test users
    test_users = users.sample(min(args.test_users, len(users)))["user_id"].tolist()
    
    # Compare models
    results = compare_recommenders(models, test_users, k=args.k)
    
    # Display results
    print("\nEvaluation Results:")
    print(results)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/evaluation_results.csv", index=False)
    print("Results saved to results/evaluation_results.csv")

def run_demo(args):
    """Run interactive demo"""
    demo = SustainableTourismDemo()
    demo.load_data()
    demo.initialize_recommender(sustainability_weight=args.weight)
    demo.run_demo()

def run_chatbot_demo(args):
    """Run interactive chatbot demo"""
    demo = SustainableTourismDemo()
    demo.load_data()
    demo.initialize_recommender(sustainability_weight=args.weight)
    demo.run_chatbot_demo()

def main():
    """Main function to parse arguments and run specified action"""
    parser = argparse.ArgumentParser(description="Sustainable Tourism Recommender System")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Generate data parser
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--destinations", type=int, default=100, help="Number of destinations")
    gen_parser.add_argument("--activities", type=int, default=300, help="Number of activities")
    gen_parser.add_argument("--users", type=int, default=1000, help="Number of users")
    
    # Process data parser
    proc_parser = subparsers.add_parser("process", help="Process raw data")
    
    # Train models parser
    train_parser = subparsers.add_parser("train", help="Train recommendation models")
    train_parser.add_argument("--neural", action="store_true", help="Include neural CF model")
    
    # Evaluate parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate recommendation models")
    eval_parser.add_argument("--test_users", type=int, default=100, help="Number of test users")
    eval_parser.add_argument("--k", type=int, default=5, help="Number of recommendations to consider")
    eval_parser.add_argument("--neural", action="store_true", help="Include neural CF model")
    
    # Demo parser
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--weight", type=float, default=0.3, help="Sustainability weight")
    
    # Chatbot parser
    chatbot_parser = subparsers.add_parser("chatbot", help="Run interactive chatbot")
    chatbot_parser.add_argument("--weight", type=float, default=0.3, help="Sustainability weight")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute specified action
    if args.action == "generate":
        generate_data(args)
    elif args.action == "process":
        process_data(args)
    elif args.action == "train":
        train_models(args)
    elif args.action == "evaluate":
        evaluate(args)
    elif args.action == "demo":
        run_demo(args)
    elif args.action == "chatbot":
        run_chatbot_demo(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()