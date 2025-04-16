from setuptools import setup, find_packages

setup(
    name="sustainable_tourism_recommender",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tensorflow>=2.4.0",
        "torch>=1.8.0",
        "transformers>=4.5.0",
        "shap>=0.39.0",
        "jupyter>=1.0.0",
        "plotly>=4.14.0",
        "networkx>=2.5.0",
        "tqdm>=4.6.0"
    ],
    author="CET313 Student",
    author_email="student@example.com",
    description="Sustainable Tourism Recommender System for CET313 AI Module",
    keywords="recommender, AI, sustainability, tourism",
    python_requires=">=3.7",
)