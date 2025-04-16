import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_data_generator import SyntheticDataGenerator
from src.data.data_processor import DataProcessor

class TestDataGeneration(unittest.TestCase):
    """Test data generation and processing"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create a small dataset for testing
        self.generator = SyntheticDataGenerator(
            num_destinations=10,
            num_activities=30,
            num_users=20
        )
    
    def test_generate_destinations(self):
        """Test destination data generation"""
        destinations = self.generator.generate_destinations()
        
        # Check that the DataFrame has the correct shape
        self.assertEqual(len(destinations), 10)
        
        # Check that required columns exist
        required_columns = ["destination_id", "name", "country", "landscape_type"]
        for col in required_columns:
            self.assertIn(col, destinations.columns)
        
        # Check that destination_id is unique
        self.assertEqual(len(destinations["destination_id"].unique()), 10)
    
    def test_generate_sustainability_metrics(self):
        """Test sustainability metrics generation"""
        destinations = self.generator.generate_destinations()
        metrics = self.generator.generate_sustainability_metrics(destinations["destination_id"].tolist())
        
        # Check that the DataFrame has the correct shape
        self.assertEqual(len(metrics), 10)
        
        # Check that required columns exist
        required_columns = ["destination_id", "carbon_footprint_score", "overall_sustainability_score"]
        for col in required_columns:
            self.assertIn(col, metrics.columns)
        
        # Check that scores are in the valid range (1-10)
        self.assertTrue(all(metrics["carbon_footprint_score"] >= 1))
        self.assertTrue(all(metrics["carbon_footprint_score"] <= 10))
        self.assertTrue(all(metrics["overall_sustainability_score"] >= 1))
        self.assertTrue(all(metrics["overall_sustainability_score"] <= 10))
    
    def test_generate_activities(self):
        """Test activity data generation"""
        destinations = self.generator.generate_destinations()
        activities = self.generator.generate_activities(destinations["destination_id"].tolist())
        
        # Check that activities were generated
        self.assertTrue(len(activities) > 0)
        
        # Check that required columns exist
        required_columns = ["activity_id", "name", "destination_id", "category"]
        for col in required_columns:
            self.assertIn(col, activities.columns)
        
        # Check that activity_id is unique
        self.assertEqual(len(activities["activity_id"].unique()), len(activities))
        
        # Check that all activities are linked to valid destinations
        self.assertTrue(all(activities["destination_id"].isin(destinations["destination_id"])))
    
    def test_generate_users(self):
        """Test user data generation"""
        users = self.generator.generate_users()
        
        # Check that the DataFrame has the correct shape
        self.assertEqual(len(users), 20)
        
        # Check that required columns exist
        required_columns = ["user_id", "age_group", "interests", "sustainability_preference"]
        for col in required_columns:
            self.assertIn(col, users.columns)
        
        # Check that user_id is unique
        self.assertEqual(len(users["user_id"].unique()), 20)
        
        # Check that sustainability_preference is in the valid range (1-10)
        self.assertTrue(all(users["sustainability_preference"] >= 1))
        self.assertTrue(all(users["sustainability_preference"] <= 10))
    
    def test_generate_all_data(self):
        """Test generating all datasets"""
        datasets = self.generator.generate_all_data()
        
        # Check that all datasets were generated
        self.assertIn("destinations", datasets)
        self.assertIn("sustainability_metrics", datasets)
        self.assertIn("activities", datasets)
        self.assertIn("users", datasets)
        
        # Check that the datasets have the correct shapes
        self.assertEqual(len(datasets["destinations"]), 10)
        self.assertEqual(len(datasets["sustainability_metrics"]), 10)
        self.assertTrue(len(datasets["activities"]) > 0)
        self.assertEqual(len(datasets["users"]), 20)
    
    def test_save_data(self):
        """Test saving data to files"""
        # Generate and save data
        self.generator.save_data(data_dir=self.test_data_dir)
        
        # Check that files were created
        self.assertTrue(os.path.exists(f"{self.test_data_dir}/destinations.csv"))
        self.assertTrue(os.path.exists(f"{self.test_data_dir}/sustainability_metrics.csv"))
        self.assertTrue(os.path.exists(f"{self.test_data_dir}/activities.csv"))
        self.assertTrue(os.path.exists(f"{self.test_data_dir}/users.csv"))
        
        # Clean up
        for file in ["destinations.csv", "sustainability_metrics.csv", "activities.csv", "users.csv"]:
            os.remove(f"{self.test_data_dir}/{file}")
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test directory if it exists
        if os.path.exists(self.test_data_dir):
            import shutil
            shutil.rmtree(self.test_data_dir)


class TestDataProcessing(unittest.TestCase):
    """Test data processing"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for test data
        self.test_raw_dir = "test_data/raw"
        self.test_processed_dir = "test_data/processed"
        os.makedirs(self.test_raw_dir, exist_ok=True)
        os.makedirs(self.test_processed_dir, exist_ok=True)
        
        # Generate test data
        self.generator = SyntheticDataGenerator(
            num_destinations=10,
            num_activities=30,
            num_users=20
        )
        self.generator.save_data(data_dir=self.test_raw_dir)
        
        # Create processor
        self.processor = DataProcessor(data_dir=self.test_raw_dir)
        self.processor.processed_dir = self.test_processed_dir
    
    def test_load_data(self):
        """Test loading data"""
        self.processor.load_data()
        
        # Check that data was loaded
        self.assertIsNotNone(self.processor.destinations)
        self.assertIsNotNone(self.processor.sustainability)
        self.assertIsNotNone(self.processor.activities)
        self.assertIsNotNone(self.processor.users)
        
        # Check that the data has the correct shapes
        self.assertEqual(len(self.processor.destinations), 10)
        self.assertEqual(len(self.processor.sustainability), 10)
        self.assertTrue(len(self.processor.activities) > 0)
        self.assertEqual(len(self.processor.users), 20)
    
    def test_preprocess_destinations(self):
        """Test preprocessing destinations"""
        self.processor.load_data()
        processed = self.processor.preprocess_destinations()
        
        # Check that destinations were merged with sustainability metrics
        self.assertIn("carbon_footprint_score", processed.columns)
        self.assertIn("overall_sustainability_score", processed.columns)
        
        # Check that categorical columns were converted
        self.assertEqual(processed["country"].dtype.name, "category")
        self.assertEqual(processed["climate"].dtype.name, "category")
        
        # Check that list columns were processed
        list_cols = ["popular_activities", "accommodation_types"]
        for col in list_cols:
            if col in processed.columns:
                self.assertTrue(isinstance(processed[col].iloc[0], list))
    
    def test_preprocess_activities(self):
        """Test preprocessing activities"""
        self.processor.load_data()
        processed = self.processor.preprocess_activities()
        
        # Check that category was converted
        self.assertEqual(processed["category"].dtype.name, "category")
        
        # Check that season encoding was added
        self.assertIn("season_code", processed.columns)
        
        # Check that eco-friendliness score was calculated
        self.assertIn("eco_friendliness", processed.columns)
        self.assertTrue(all(processed["eco_friendliness"] >= 0))
        self.assertTrue(all(processed["eco_friendliness"] <= 10))
    
    def test_preprocess_users(self):
        """Test preprocessing users"""
        self.processor.load_data()
        processed = self.processor.preprocess_users()
        
        # Check that list columns were processed
        list_cols = ["interests", "travel_history", "preferred_activities"]
        for col in list_cols:
            self.assertTrue(isinstance(processed[col].iloc[0], list))
        
        # Check that sustainability group was added
        self.assertIn("sustainability_group", processed.columns)
    
    def test_create_user_destination_matrix(self):
        """Test creating user-destination interaction matrix"""
        self.processor.load_data()
        self.processor.preprocess_users()
        
        matrix, user_ids, dest_ids = self.processor.create_user_destination_matrix()
        
        # Check matrix shape
        self.assertEqual(matrix.shape, (20, 10))  # 20 users, 10 destinations
        
        # Check that matrix contains binary values
        self.assertTrue(np.all((matrix == 0) | (matrix == 1)))
        
        # Check that user and destination IDs match matrix dimensions
        self.assertEqual(len(user_ids), 20)
        self.assertEqual(len(dest_ids), 10)
    
    def test_process_all(self):
        """Test processing all data"""
        processed_data = self.processor.process_all()
        
        # Check that all processed data was returned
        self.assertIn("destinations", processed_data)
        self.assertIn("activities", processed_data)
        self.assertIn("users", processed_data)
        self.assertIn("interaction_matrix", processed_data)
        
        # Check that processed files were created
        self.assertTrue(os.path.exists(f"{self.test_processed_dir}/destinations.pkl"))
        self.assertTrue(os.path.exists(f"{self.test_processed_dir}/activities.pkl"))
        self.assertTrue(os.path.exists(f"{self.test_processed_dir}/users.pkl"))
        self.assertTrue(os.path.exists(f"{self.test_processed_dir}/interaction_matrix.pkl"))
        self.assertTrue(os.path.exists(f"{self.test_processed_dir}/destination_features.pkl"))
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test directories if they exist
        if os.path.exists("test_data"):
            import shutil
            shutil.rmtree("test_data")


if __name__ == '__main__':
    unittest.main()