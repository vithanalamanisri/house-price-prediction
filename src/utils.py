import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

def create_sample_data(n_samples=2000):
    """Create realistic sample dataset"""
    np.random.seed(42)
    
    # Generate features
    area = np.random.normal(2000, 800, n_samples)
    area = np.clip(area, 500, 5000).astype(int)
    
    bedrooms = np.round(area / 600 + np.random.normal(0, 0.5, n_samples))
    bedrooms = np.clip(bedrooms, 1, 6).astype(int)
    
    bathrooms = bedrooms + np.random.uniform(-0.5, 1, n_samples)
    bathrooms = np.clip(bathrooms, 1, 5).round(1)
    
    age = np.random.exponential(15, n_samples)
    age = np.clip(age, 0, 60).astype(int)
    
    location_score = np.random.beta(2, 2, n_samples) * 9 + 1
    location_score = location_score.round(1)
    
    garage_spaces = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.6, 0.2])
    has_basement = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    has_pool = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    # Calculate price with realistic relationships
    price = (50000 +
             area * 120 +
             bedrooms * 8000 +
             bathrooms * 10000 -
             age * 800 +
             location_score * 30000 +
             garage_spaces * 15000 +
             has_basement * 20000 +
             has_pool * 25000 +
             np.random.normal(0, 30000, n_samples))
    
    price = np.clip(price, 50000, 1000000).astype(int)
    
    df = pd.DataFrame({
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'garage_spaces': garage_spaces,
        'has_basement': has_basement,
        'has_pool': has_pool,
        'price': price
    })
    
    return df

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MSE': float(mean_squared_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    }

def save_metrics(metrics, filename='models/model_metrics.json'):
    """Save metrics to JSON file"""
    os.makedirs('models', exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {}
    for model_name, model_metrics in metrics.items():
        metrics_serializable[model_name] = {}
        for metric_name, metric_value in model_metrics.items():
            if hasattr(metric_value, 'item'):  # Check if it's numpy type
                metrics_serializable[model_name][metric_name] = float(metric_value)
            else:
                metrics_serializable[model_name][metric_name] = metric_value
    
    with open(filename, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_imp, x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        return fig
    return None