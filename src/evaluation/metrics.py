import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.base_models import BaseRecommender

def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """
    Calculate precision@k
    
    Parameters:
    - y_true: Binary array of true values
    - y_pred: Array of predicted values/scores
    - k: Number of top items to consider
    
    Returns:
    - Precision@k score
    """
    if len(y_true) == 0 or k == 0:
        return 0.0
    
    # Get indices of top-k predictions
    top_k_indices = np.argsort(-y_pred)[:k]
    
    # Calculate precision
    hits = np.sum(y_true[top_k_indices])
    return hits / k

def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """
    Calculate recall@k
    
    Parameters:
    - y_true: Binary array of true values
    - y_pred: Array of predicted values/scores
    - k: Number of top items to consider
    
    Returns:
    - Recall@k score
    """
    if len(y_true) == 0 or np.sum(y_true) == 0 or k == 0:
        return 0.0
    
    # Get indices of top-k predictions
    top_k_indices = np.argsort(-y_pred)[:k]
    
    # Calculate recall
    hits = np.sum(y_true[top_k_indices])
    return hits / np.sum(y_true)

def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k
    
    Parameters:
    - y_true: Binary array of true values
    - y_pred: Array of predicted values/scores
    - k: Number of top items to consider
    
    Returns:
    - NDCG@k score
    """
    if len(y_true) == 0 or np.sum(y_true) == 0 or k == 0:
        return 0.0
    
    # Get indices of top-k predictions
    top_k_indices = np.argsort(-y_pred)[:k]
    
    # Calculate DCG
    dcg = np.sum(y_true[top_k_indices] / np.log2(np.arange(2, k + 2)))
    
    # Calculate ideal DCG (IDCG)
    ideal_indices = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_indices] / np.log2(np.arange(2, k + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def sustainability_score(recommendations: List[Dict[str, Any]]) -> float:
    """
    Calculate average sustainability score of recommendations
    
    Parameters:
    - recommendations: List of recommendation dictionaries
    
    Returns:
    - Average sustainability score (0-10)
    """
    if not recommendations:
        return 0.0
    
    scores = [rec["sustainability_score"] for rec in recommendations if "sustainability_score" in rec]
    
    if not scores:
        return 0.0
    
    return np.mean(scores)

def evaluate_recommender(recommender: BaseRecommender, test_users: List[int], 
                        k: int = 5) -> Dict[str, float]:
    """
    Evaluate a recommender model using various metrics
    
    Parameters:
    - recommender: Trained recommender model
    - test_users: List of user IDs for testing
    - k: Number of recommendations to consider
    
    Returns:
    - Dictionary of evaluation metrics
    """
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    sustainability_scores = []
    
    for user_id in test_users:
        # Get user index
        user_idx = recommender._get_user_index(user_id)
        
        # Split user's data into train and test
        n_interactions = recommender.interaction_matrix.shape[1]
        
        # Mask 20% of the user's positive interactions for testing
        positive_indices = np.where(recommender.interaction_matrix[user_idx] > 0)[0]
        
        if len(positive_indices) < 5:  # Skip users with too few interactions
            continue
        
        # Create a mask to hide 20% of positive interactions
        n_test = max(1, int(0.2 * len(positive_indices)))
        np.random.shuffle(positive_indices)
        test_indices = positive_indices[:n_test]
        
        # Create temporary matrices for training and testing
        train_matrix = recommender.interaction_matrix.copy()
        test_matrix = np.zeros_like(recommender.interaction_matrix)
        
        # Hide test interactions from training data
        for idx in test_indices:
            train_matrix[user_idx, idx] = 0
            test_matrix[user_idx, idx] = 1
        
        # Store original matrix
        original_matrix = recommender.interaction_matrix.copy()
        
        # Temporarily replace the interaction matrix
        recommender.interaction_matrix = train_matrix
        
        # Get recommendations
        recs = recommender.recommend(user_id, n=k, exclude_visited=True)
        
        # Convert recommendations to scores vector
        scores = np.zeros(n_interactions)
        for i, rec in enumerate(recs):
            dest_idx = recommender._get_destination_index(rec["destination_id"])
            scores[dest_idx] = n_interactions - i  # Higher score for higher ranked items
        
        # Calculate metrics
        y_true = test_matrix[user_idx]
        precision_scores.append(precision_at_k(y_true, scores, k))
        recall_scores.append(recall_at_k(y_true, scores, k))
        ndcg_scores.append(ndcg_at_k(y_true, scores, k))
        sustainability_scores.append(sustainability_score(recs))
        
        # Restore original matrix
        recommender.interaction_matrix = original_matrix
    
    # Calculate average metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_sustainability = np.mean(sustainability_scores) if sustainability_scores else 0.0
    
    return {
        f"precision@{k}": avg_precision,
        f"recall@{k}": avg_recall,
        f"ndcg@{k}": avg_ndcg,
        "sustainability_score": avg_sustainability
    }

def compare_recommenders(recommenders: List[BaseRecommender], test_users: List[int], 
                         k: int = 5) -> pd.DataFrame:
    """
    Compare multiple recommender models using various metrics
    
    Parameters:
    - recommenders: List of trained recommender models
    - test_users: List of user IDs for testing
    - k: Number of recommendations to consider
    
    Returns:
    - DataFrame with comparison metrics
    """
    results = []
    
    for recommender in recommenders:
        print(f"Evaluating {recommender.name}...")
        metrics = evaluate_recommender(recommender, test_users, k)
        metrics["model"] = recommender.name
        results.append(metrics)
    
    return pd.DataFrame(results)

def diversity_score(recommendations: List[Dict[str, Any]]) -> float:
    """
    Calculate diversity score of recommendations (based on unique countries)
    
    Parameters:
    - recommendations: List of recommendation dictionaries
    
    Returns:
    - Diversity score (0-1)
    """
    if not recommendations:
        return 0.0
    
    countries = [rec["country"] for rec in recommendations if "country" in rec]
    unique_countries = len(set(countries))
    
    return unique_countries / len(recommendations)