import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_data_generator import SyntheticDataGenerator
from src.data.data_processor import DataProcessor
from src.sustainability.sustainability_scorer import SustainabilityScorer
from src.sustainability.impact_calculator import EnvironmentalImpactCalculator
from src.sustainability.weighting import SustainabilityWeighting

class TestSustainabilityComponents(unittest.TestCase):
    """Test sustainability-related components"""
    
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
        self.activities = pd.read_pickle(f"{self.test_processed_dir}/activities.pkl")
    
    def test_sustainability_scorer(self):
        """Test sustainability scorer"""
        # Create scorer
        scorer = SustainabilityScorer(self.destinations)
        
        # Test getting sustainability score
        test_dest_id = self.destinations["destination_id"].iloc[0]
        score = scorer.get_sustainability_score(test_dest_id)
        
        # Check that score is in valid range
        self.assertTrue(0 <= score <= 10)
        
        # Test getting detailed scores
        detailed_scores = scorer.get_detailed_sustainability_scores(test_dest_id)
        
        # Check that detailed scores contain expected metrics
        expected_metrics = [
            "carbon_footprint_score",
            "water_consumption_score",
            "waste_management_score",
            "biodiversity_impact_score",
            "local_economy_support_score",
            "overall"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, detailed_scores)
            self.assertTrue(0 <= detailed_scores[metric] <= 10)
    
    def test_apply_sustainability_weighting(self):
        """Test applying sustainability weighting to recommendations"""
        # Create scorer
        scorer = SustainabilityScorer(self.destinations)
        
        # Create sample recommendations
        recs = []
        for i in range(5):
            dest_id = self.destinations["destination_id"].iloc[i]
            dest = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
            
            recs.append({
                "destination_id": int(dest_id),
                "name": dest["name"],
                "country": dest["country"],
                "sustainability_score": dest["overall_sustainability_score"]
            })
        
        # Apply weighting with different weights
        for weight in [0.0, 0.5, 1.0]:
            weighted_recs = scorer.apply_sustainability_weighting(recs, weight=weight)
            
            # Check that weighted recommendations have the same items
            self.assertEqual(len(weighted_recs), len(recs))
            self.assertEqual(set(r["destination_id"] for r in weighted_recs),
                            set(r["destination_id"] for r in recs))
            
            # Check that order changes with weight
            if weight > 0:
                # With weight > 0, order should be influenced by sustainability
                ordered_dest_ids = [r["destination_id"] for r in weighted_recs]
                ordered_scores = [r["sustainability_score"] for r in weighted_recs]
                
                # Check if scores are in descending order (at least partially)
                # This is not a strict sort since original order also matters
                has_influence = False
                for i in range(len(ordered_scores) - 1):
                    if ordered_scores[i] < ordered_scores[i + 1]:
                        has_influence = True
                        break
                
                if weight == 1.0:
                    # With weight = 1.0, order should match sustainability scores exactly
                    self.assertEqual(ordered_scores, sorted(ordered_scores, reverse=True))
    
    def test_filter_by_sustainability_threshold(self):
        """Test filtering recommendations by sustainability threshold"""
        # Create scorer
        scorer = SustainabilityScorer(self.destinations)
        
        # Create sample recommendations
        recs = []
        for i in range(10):
            dest_id = self.destinations["destination_id"].iloc[i]
            dest = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
            
            recs.append({
                "destination_id": int(dest_id),
                "name": dest["name"],
                "country": dest["country"],
                "sustainability_score": dest["overall_sustainability_score"]
            })
        
        # Apply threshold filtering
        threshold = 7.0
        filtered_recs = scorer.filter_by_sustainability_threshold(recs, threshold=threshold)
        
        # Check that filtered recommendations meet threshold
        for rec in filtered_recs:
            self.assertTrue(rec["sustainability_score"] >= threshold)
    
    def test_environmental_impact_calculator(self):
        """Test environmental impact calculator"""
        # Create calculator
        calculator = EnvironmentalImpactCalculator(self.destinations, self.activities)
        
        # Test calculating destination impact
        test_dest_id = self.destinations["destination_id"].iloc[0]
        impact = calculator.calculate_destination_impact(test_dest_id)
        
        # Check that impact contains expected metrics
        expected_metrics = [
            "carbon_impact",
            "water_impact",
            "waste_impact",
            "biodiversity_impact",
            "total_impact_score",
            "impact_percentage"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, impact)
            self.assertTrue(isinstance(impact[metric], (int, float)))
        
        # Test calculating activity impact
        test_act_id = self.activities["activity_id"].iloc[0]
        act_impact = calculator.calculate_activity_impact(test_act_id)
        
        # Check that activity impact contains expected metrics
        expected_act_metrics = [
            "environmental_impact",
            "resource_consumption",
            "community_benefit",
            "net_impact",
            "scaled_impact"
        ]
        
        for metric in expected_act_metrics:
            self.assertIn(metric, act_impact)
    
    def test_sustainability_weighting(self):
        """Test sustainability weighting schemes"""
        # Create weighting
        weighting = SustainabilityWeighting()
        
        # Test different weighting schemes
        base_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        sustainability_scores = [0.5, 0.7, 0.9, 0.3, 0.6]
        
        for scheme in ["linear", "quadratic", "sigmoid", "threshold"]:
            weighted_scores = weighting.apply_weighting(
                base_scores, sustainability_scores, scheme=scheme, weight=0.5
            )
            
            # Check that weighted scores are valid
            self.assertEqual(len(weighted_scores), len(base_scores))
            self.assertTrue(all(0 <= score <= 1 for score in weighted_scores))
            
            # Check that weights influence the scores
            for i in range(len(base_scores)):
                self.assertNotEqual(weighted_scores[i], base_scores[i])
    
    def test_apply_weighting_to_recommendations(self):
        """Test applying weighting schemes to recommendations"""
        # Create weighting
        weighting = SustainabilityWeighting()
        
        # Create sample recommendations
        recs = []
        for i in range(5):
            dest_id = self.destinations["destination_id"].iloc[i]
            dest = self.destinations[self.destinations["destination_id"] == dest_id].iloc[0]
            
            recs.append({
                "destination_id": int(dest_id),
                "name": dest["name"],
                "country": dest["country"],
                "sustainability_score": dest["overall_sustainability_score"]
            })
        
        # Apply weighting to recommendations
        for scheme in ["linear", "quadratic", "sigmoid", "threshold"]:
            weighted_recs = weighting.apply_weighting_to_recommendations(
                recs, scheme=scheme, weight=0.5
            )
            
            # Check that weighted recommendations are valid
            self.assertEqual(len(weighted_recs), len(recs))
            
            # Check that destination IDs remain the same
            self.assertEqual(set(r["destination_id"] for r in weighted_recs),
                            set(r["destination_id"] for r in recs))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test directories if they exist
        if os.path.exists("test_data"):
            import shutil
            shutil.rmtree("test_data")


if __name__ == '__main__':
    unittest.main()