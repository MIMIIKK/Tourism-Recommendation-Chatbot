import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple, Any
import pickle
import os
from src.models.base_models import BaseRecommender

class NeuralCollaborativeFiltering(BaseRecommender):
    """Neural Collaborative Filtering model for recommendations"""
    
    def __init__(self, embedding_dim: int = 32, hidden_layers: List[int] = [64, 32]):
        super().__init__(name="NeuralCF")
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.user_embedding = None
        self.dest_embedding = None
        self.num_users = None
        self.num_destinations = None
    
    def build_model(self):
        """Build the neural collaborative filtering model"""
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        dest_input = Input(shape=(1,), name='dest_input')
        
        # Embedding layers
        self.user_embedding = Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        
        self.dest_embedding = Embedding(
            input_dim=self.num_destinations,
            output_dim=self.embedding_dim,
            name='dest_embedding'
        )(dest_input)
        
        # Flatten embeddings
        user_vec = Flatten()(self.user_embedding)
        dest_vec = Flatten()(self.dest_embedding)
        
        # Concatenate embeddings
        concat = Concatenate()([user_vec, dest_vec])
        
        # Hidden layers
        x = concat
        for i, units in enumerate(self.hidden_layers):
            x = Dense(units, activation='relu', name=f'hidden_{i+1}')(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create and compile model
        self.model = Model(inputs=[user_input, dest_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def prepare_training_data(self):
        """Prepare training data from interaction matrix"""
        user_indices = []
        dest_indices = []
        labels = []
        
        # Convert interaction matrix to training data
        for user_idx in range(self.interaction_matrix.shape[0]):
            for dest_idx in range(self.interaction_matrix.shape[1]):
                user_indices.append(user_idx)
                dest_indices.append(dest_idx)
                labels.append(self.interaction_matrix[user_idx, dest_idx])
        
        return np.array(user_indices), np.array(dest_indices), np.array(labels)
    
    def fit(self, epochs: int = 20, batch_size: int = 64):
        """Train the neural collaborative filtering model"""
        self.num_users = len(self.user_ids)
        self.num_destinations = len(self.dest_ids)
        
        # Build the model
        self.build_model()
        
        # Prepare training data
        user_indices, dest_indices, labels = self.prepare_training_data()
        
        # Train the model
        history = self.model.fit(
            [user_indices, dest_indices],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def recommend(self, user_id: int, n: int = 5, exclude_visited: bool = True) -> List[Dict[str, Any]]:
        """Generate recommendations using the trained neural CF model"""
        user_idx = self._get_user_index(user_id)
        
        # Create input arrays for prediction
        user_input = np.array([user_idx] * len(self.dest_ids))
        dest_input = np.array(range(len(self.dest_ids)))
        
        # Predict scores for all destinations
        scores = self.model.predict([user_input, dest_input], verbose=0).flatten()
        
        # Sort destinations by score
        dest_indices = np.argsort(-scores)
        
        if exclude_visited:
            # Filter out destinations the user has already visited
            visited = self.interaction_matrix[user_idx] > 0
            dest_indices = dest_indices[~visited[dest_indices]]
        
        # Return top n recommendations
        top_n_indices = dest_indices[:n]
        return self._format_recommendations(top_n_indices)
    
    def save_model(self, model_dir: str = "models"):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(f"{model_dir}/neural_cf_model")
        
        # Save model metadata
        with open(f"{model_dir}/neural_cf_metadata.pkl", "wb") as f:
            pickle.dump({
                "num_users": self.num_users,
                "num_destinations": self.num_destinations,
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers
            }, f)
    
    def load_model(self, model_dir: str = "models"):
        """Load a trained model"""
        # Load model metadata
        with open(f"{model_dir}/neural_cf_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.num_users = metadata["num_users"]
            self.num_destinations = metadata["num_destinations"]
            self.embedding_dim = metadata["embedding_dim"]
            self.hidden_layers = metadata["hidden_layers"]
        
        # Load the model
        self.model = tf.keras.models.load_model(f"{model_dir}/neural_cf_model")
        
        return self