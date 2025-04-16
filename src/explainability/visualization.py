import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import os

class ExplanationVisualizer:
    """Class for creating visualizations to explain recommendations"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set visualization style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_sustainability_metrics(self, metrics: Dict[str, float], 
                                  destination_name: str = "Destination",
                                  save_path: str = None) -> str:
        """
        Plot sustainability metrics for a destination
        
        Parameters:
        - metrics: Dictionary of sustainability metrics
        - destination_name: Name of the destination
        - save_path: Path to save the plot (if None, will generate a default path)
        
        Returns:
        - Path to the saved plot
        """
        # Create a DataFrame from metrics
        df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Score': list(metrics.values())
        })
        
        # Sort by score
        df = df.sort_values('Score', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Score', y='Metric', data=df, palette='YlGnBu')
        
        # Add value labels
        for i, score in enumerate(df['Score']):
            ax.text(score + 0.1, i, f"{score:.1f}", va='center')
        
        # Set limits and title
        plt.xlim(0, 10.5)
        plt.title(f"Sustainability Metrics for {destination_name}")
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            dest_name_safe = destination_name.replace(" ", "_").lower()
            save_path = os.path.join(self.save_dir, f"sustainability_{dest_name_safe}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_sustainability_comparison(self, destinations: List[Dict[str, Any]], 
                                     metrics: List[str] = None,
                                     save_path: str = None) -> str:
        """
        Plot sustainability comparison between multiple destinations
        
        Parameters:
        - destinations: List of destination dictionaries with metrics
        - metrics: List of metrics to compare (if None, will use all common metrics)
        - save_path: Path to save the plot
        
        Returns:
        - Path to the saved plot
        """
        if not destinations:
            raise ValueError("No destinations provided for comparison")
        
        # If metrics not specified, use common metrics from the first destination
        if metrics is None and "metrics" in destinations[0]:
            metrics = list(destinations[0]["metrics"].keys())
        elif metrics is None:
            metrics = ["overall_sustainability_score"]
        
        # Create a DataFrame for plotting
        data = []
        
        for dest in destinations:
            dest_name = dest.get("name", "Unknown")
            
            for metric in metrics:
                # Handle different data structures
                if "metrics" in dest and metric in dest["metrics"]:
                    score = dest["metrics"][metric]
                elif metric in dest:
                    score = dest[metric]
                else:
                    score = None
                
                if score is not None:
                    data.append({
                        "Destination": dest_name,
                        "Metric": metric.replace("_", " ").title(),
                        "Score": score
                    })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise ValueError("No valid data for comparison plot")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Destination', y='Score', hue='Metric', data=df, palette='viridis')
        
        # Add legend and title
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Sustainability Comparison Between Destinations')
        plt.ylim(0, 10.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.save_dir, "sustainability_comparison.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_counterfactual_explanation(self, explanation: Dict[str, Any],
                                      save_path: str = None) -> str:
        """
        Plot a visualization of a counterfactual explanation
        
        Parameters:
        - explanation: Counterfactual explanation dictionary
        - save_path: Path to save the plot
        
        Returns:
        - Path to the saved plot
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data from explanation
        dest_name = explanation.get("destination_name", "Destination")
        current_rank = explanation.get("current_rank")
        counterfactual_rank = explanation.get("counterfactual_rank")
        
        if current_rank is None or counterfactual_rank == "Not in top 20":
            # Can't create rank comparison plot
            feature = explanation.get("feature", explanation.get("user_feature", "sustainability_weight"))
            current_value = explanation.get("current_value", explanation.get("current_weight"))
            target_value = explanation.get("counterfactual_value", explanation.get("counterfactual_weight"))
            
            # Create a simple before/after plot
            plt.bar([0, 1], [current_value, target_value], color=['skyblue', 'coral'])
            plt.xticks([0, 1], ['Current', 'Counterfactual'])
            plt.ylabel('Value')
            plt.title(f"Counterfactual Analysis: Changing {feature.replace('_', ' ').title()}\nfor {dest_name}")
            
        else:
            # Convert to int if possible
            if isinstance(counterfactual_rank, str) and counterfactual_rank.isdigit():
                counterfactual_rank = int(counterfactual_rank)
            
            # Create a rank comparison plot
            ranks = [current_rank, counterfactual_rank]
            plt.bar([0, 1], ranks, color=['skyblue', 'coral'])
            plt.xticks([0, 1], ['Current', 'Counterfactual'])
            plt.ylabel('Rank')
            plt.gca().invert_yaxis()  # Invert so lower ranks (better) are higher on the chart
            
            feature = explanation.get("feature", explanation.get("user_feature", "sustainability_weight"))
            current_value = explanation.get("current_value", explanation.get("current_weight"))
            target_value = explanation.get("counterfactual_value", explanation.get("counterfactual_weight"))
            
            plt.title(f"Counterfactual Analysis: Rank Change for {dest_name}\nwhen {feature.replace('_', ' ').title()} changes from {current_value} to {target_value}")
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            dest_name_safe = dest_name.replace(" ", "_").lower()
            feature_safe = explanation.get("feature", "sustainability").replace(" ", "_").lower()
            save_path = os.path.join(self.save_dir, f"counterfactual_{dest_name_safe}_{feature_safe}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_recommendation_sources(self, recommendation: Dict[str, Any],
                                  sources: List[Tuple[str, float]],
                                  save_path: str = None) -> str:
        """
        Plot the contribution of different recommendation sources
        
        Parameters:
        - recommendation: Recommendation dictionary
        - sources: List of (source_name, contribution_score) tuples
        - save_path: Path to save the plot
        
        Returns:
        - Path to the saved plot
        """
        # Sort sources by contribution
        sources = sorted(sources, key=lambda x: x[1])
        
        # Extract data
        source_names = [s[0] for s in sources]
        contributions = [s[1] for s in sources]
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(source_names, contributions, color=sns.color_palette("viridis", len(sources)))
        
        # Add labels and title
        dest_name = recommendation.get("name", "Destination")
        plt.xlabel('Contribution Score')
        plt.title(f"Recommendation Sources for {dest_name}")
        plt.xlim(0, max(contributions) * 1.1)
        
        # Add value labels
        for i, v in enumerate(contributions):
            plt.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            dest_name_safe = dest_name.replace(" ", "_").lower()
            save_path = os.path.join(self.save_dir, f"sources_{dest_name_safe}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_sustainability_impact(self, recommendations: List[Dict[str, Any]],
                                 with_weighting: List[Dict[str, Any]],
                                 save_path: str = None) -> str:
        """
        Plot the impact of sustainability weighting on recommendations
        
        Parameters:
        - recommendations: List of recommendations without sustainability weighting
        - with_weighting: List of recommendations with sustainability weighting
        - save_path: Path to save the plot
        
        Returns:
        - Path to the saved plot
        """
        # Calculate average sustainability scores
        avg_without = np.mean([r["sustainability_score"] for r in recommendations])
        avg_with = np.mean([r["sustainability_score"] for r in with_weighting])
        
        # Create comparison data
        data = []
        
        # Add top 3 recommendations from each approach
        for i, rec in enumerate(recommendations[:3]):
            data.append({
                "Approach": "Without Weighting",
                "Rank": i + 1,
                "Destination": rec["name"],
                "Sustainability Score": rec["sustainability_score"]
            })
        
        for i, rec in enumerate(with_weighting[:3]):
            data.append({
                "Approach": "With Weighting",
                "Rank": i + 1,
                "Destination": rec["name"],
                "Sustainability Score": rec["sustainability_score"]
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Bar plot for top recommendations
        sns.barplot(x='Rank', y='Sustainability Score', hue='Approach', data=df, palette=['lightgray', 'seagreen'])
        
        # Add average lines
        plt.axhline(y=avg_without, color='gray', linestyle='--', label=f'Avg. Without: {avg_without:.1f}')
        plt.axhline(y=avg_with, color='green', linestyle='--', label=f'Avg. With: {avg_with:.1f}')
        
        # Add labels and title
        plt.title('Impact of Sustainability Weighting on Recommendations')
        plt.ylim(0, 10)
        plt.legend(title='')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.save_dir, "sustainability_impact.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path