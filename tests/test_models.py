import unittest
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_data_generator import SyntheticDataGenerator
from src.data.data_processor import DataProcessor
from src.models.base_models import (
    BaseRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender
)
from src.models.ensemble import HybridRecommender, create_default_hybrid_recommender

class TestRecommendationModels(unittest.TestCase):
    """Test recommendation models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Create temporary directories for test data
        cls.test_raw_dir = "test_data/raw"
        cls.test_processed_dir = "test_data/processed"
        os.makedirs(cls.test_raw_dir, exist_ok=True)
        os.makedirs(cls.test_processed_dir, exist_ok=True)
        
        # Generate and process test data
        generator = SyntheticDataGenerator(
            num_destinations=20,
            num_activities=60,
            num_users=50
        )
        generator.save_data(data_dir=cls.test_raw_dir)
        
        processor = DataProcessor(data_dir=cls.test_raw_dir)
        processor.processed_dir = cls.test_processed_dir
        processor.process_all()
    
    def setUp(self):
        """Set up for each test"""
        # Load processed data for verification
        self.destinations = pd.read_pickle(f"{self.test_processed_dir}/destinations.pkl")
        self.users = pd.read_pickle(f"{self.test_processed_dir}/users.pkl")
        
        with open(f"{self.test_processed_dir}/interaction_matrix.pkl", "rb") as f:
            data = pickle.load(f)
            self.interaction_matrix = data["matrix"]
            self.user_ids = data["user_ids"]
            self.dest_ids = data["dest_ids"]
    
    def test_popularity_recommender(self):
        """Test popularity-based recommender"""
        # Create and train recommender
        recommender = PopularityRecommender()
        recommender.load_data(self.test_processed_dir)
        recommender.fit()
        
        # Check that popularity scores were calculated
        self.assertIsNotNone(recommender.popularity_scores)
        self.assertEqual(len(recommender.popularity_scores), len(self.dest_ids))
        
        # Get recommendations for a user
        test_user_id = self.user_ids[0]
        recs = recommender.recommend(test_user_id, n=5)
        
        # Check recommendation format
        self.assertEqual(len(recs), 5)
        self.assertIn("destination_id", recs[0])
        self.assertIn("name", recs[0])
        self.assertIn("sustainability_score", recs[0])
        
        # Check that recommendations are sorted by popularity
        pop_scores = [np.sum(self.interaction_matrix[:, recommender._get_destination_index(rec["destination_id"])]) 
                     for rec in recs]
        self.assertEqual(pop_scores, sorted(pop_scores, reverse=True))
    
    def test_content_based_recommender(self):
        """Test content-based recommender"""
        # Create and train recommender
        recommender = ContentBasedRecommender()
        recommender.load_data(self.test_processed_dir)
        recommender.fit()
        
        # Check that similarity matrix was calculated
        self.assertIsNotNone(recommender.feature_similarity)
        self.assertEqual(recommender.feature_similarity.shape, (len(self.dest_ids), len(self.dest_ids)))
        
        # Find a user with travel history
        test_user_id = None
        for user_id in self.user_ids:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            if np.sum(self.interaction_matrix[user_idx]) > 0:
                test_user_id = user_id
                break
        
        if test_user_id is not None:
            # Get recommendations for a user with history
            recs = recommender.recommend(test_user_id, n=5)
            
            # Check recommendation format
            self.assertEqual(len(recs), 5)
            self.assertIn("destination_id", recs[0])
            self.assertIn("name", recs[0])
            
            # Check that recommendations don't include visited destinations
            user_idx = recommender._get_user_index(test_user_id)
            visited_dest_indices = np.where(recommender.interaction_matrix[user_idx] > 0)[0]
            visited_dest_ids = [recommender._get_destination_id(idx) for idx in visited_dest_indices]
            
            for rec in recs:
                self.assertNotIn(rec["destination_id"], visited_dest_ids)
    
    def test_collaborative_filtering_recommender(self):
        """Test collaborative filtering recommender"""
        # Test both user-based and item-based CF
        for method in ["user", "item"]:
            # Create and train recommender
            recommender = CollaborativeFilteringRecommender(method=method)
            recommender.load_data(self.test_processed_dir)
            recommender.fit()
            
            # Check that similarity matrix was calculated
            self.assertIsNotNone(recommender.similarity_matrix)
            
            if method == "user":
                expected_shape = (len(self.user_ids), len(self.user_ids))
            else:
                expected_shape = (len(self.dest_ids), len(self.dest_ids))
            
            self.assertEqual(recommender.similarity_matrix.shape, expected_shape)
            
            # Get recommendations for a user
            test_user_id = self.user_ids[0]
            recs = recommender.recommend(test_user_id, n=5)
            
            # Check recommendation format
            self.assertEqual(len(recs), 5)
            self.assertIn("destination_id", recs[0])
            self.assertIn("name", recs[0])
    
    def test_hybrid_recommender(self):
        """Test hybrid recommender"""
        # Create individual recommenders
        popularity_rec = PopularityRecommender()
        content_rec = ContentBasedRecommender()
        
        # Create hybrid recommender
        hybrid = HybridRecommender()
        hybrid.add_recommender(popularity_rec, weight=0.3)
        hybrid.add_recommender(content_rec, weight=0.7)
        
        # Load data and train
        hybrid.load_data(self.test_processed_dir)
        hybrid.fit()
        
        # Check that weights are normalized
        self.assertAlmostEqual(sum(hybrid.weights), 1.0)
        
        # Get recommendations
        test_user_id = self.user_ids[0]
        recs = hybrid.recommend(test_user_id, n=5)
        
        # Check recommendation format
        self.assertEqual(len(recs), 5)
        self.assertIn("destination_id", recs[0])
        self.assertIn("name", recs[0])
        self.assertIn("sustainability_score", recs[0])
    
    def test_create_default_hybrid_recommender(self):
        """Test creating default hybrid recommender"""
        # Create default hybrid recommender
        recommender = create_default_hybrid_recommender()
        
        # Check that it has the expected components
        self.assertEqual(len(recommender.recommenders), 4)
        self.assertEqual(len(recommender.weights), 4)
        
        # Load data and train
        recommender.load_data(self.test_processed_dir)
        recommender.fit()
        
        # Get recommendations
        test_user_id = self.user_ids[0]
        recs = recommender.recommend(test_user_id, n=5)
        
        # Check recommendation format
        self.assertEqual(len(recs), 5)
        self.assertIn("destination_id", recs[0])
        self.assertIn("name", recs[0])
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test directories if they exist
        if os.path.exists("test_data"):
            import shutil
            shutil.rmtree("test_data")


if __name__ == '__main__':
    unittest.main()