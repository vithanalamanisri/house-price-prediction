import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

from utils import create_sample_data, calculate_metrics, save_metrics, plot_feature_importance

def load_or_create_data():
    """Load real data or create sample data"""
    os.makedirs('data', exist_ok=True)
    
    try:
        df = pd.read_csv('data/housing.csv')
        print(f"✅ Loaded real dataset: {df.shape}")
        return df
    except:
        df = create_sample_data()
        df.to_csv('data/housing.csv', index=False)
        print(f"✅ Created sample dataset: {df.shape}")
        return df

def engineer_features(df):
    """Create advanced features"""
    df = df.copy()
    
    # Avoid division by zero
    df['bedrooms'] = df['bedrooms'].replace(0, 1)
    df['bathrooms'] = df['bathrooms'].replace(0, 1)
    
    # Ratio features
    df['area_per_bedroom'] = df['area'] / df['bedrooms']
    df['area_per_bathroom'] = df['area'] / df['bathrooms']
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
    
    # Interaction features
    df['location_age_interaction'] = df['location_score'] / (df['age'] + 1)
    df['luxury_score'] = (df['garage_spaces'] * 0.3 + 
                          df['has_basement'] * 0.3 + 
                          df['has_pool'] * 0.4)
    
    # Age categories
    df['age_category'] = pd.cut(df['age'], 
                                 bins=[-1, 5, 15, 30, 100], 
                                 labels=['New', 'Recent', 'Moderate', 'Old'])
    
    # Location categories
    df['location_category'] = pd.cut(df['location_score'],
                                      bins=[0, 3, 7, 10],
                                      labels=['Low', 'Medium', 'High'])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['age_category', 'location_category'], drop_first=True)
    
    return df

def train_multiple_models(X_train, y_train):
    """Train multiple models and find the best one"""
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'CatBoost': CatBoostRegressor(iterations=100, random_seed=42, verbose=False),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001)
    }
    
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING MULTIPLE MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Cross-validation on subset for speed
            cv_samples = min(500, len(X_train))
            cv_scores = cross_val_score(model, X_train[:cv_samples], y_train[:cv_samples], cv=3, scoring='r2')
            results[name] = {
                'model': model,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            }
            print(f"   CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
    
    return trained_models, results

def create_ensemble_model(trained_models):
    """Create a voting ensemble of top models"""
    top_models = []
    
    # Select available tree-based models
    for name in ['Random Forest', 'XGBoost', 'CatBoost', 'Gradient Boosting']:
        if name in trained_models:
            top_models.append((name.lower().replace(' ', '_'), trained_models[name]))
    
    if len(top_models) >= 2:
        ensemble = VotingRegressor(estimators=top_models)
        return ensemble
    return None

def main():
    print("🏠 ADVANCED HOUSE PRICE PREDICTION SYSTEM 🏠")
    print("="*50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = load_or_create_data()
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    df_enhanced = engineer_features(df)
    
    # Prepare features and target
    feature_cols = [col for col in df_enhanced.columns if col != 'price']
    X = df_enhanced[feature_cols]
    y = df_enhanced['price']
    
    print(f"📊 Feature count: {len(feature_cols)}")
    print(f"📊 Dataset size: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Train multiple models
    trained_models, results = train_multiple_models(X_train_scaled, y_train)
    
    if not trained_models:
        print("\n❌ No models trained successfully!")
        return None, None, None
    
    # Create ensemble
    print("\n🎯 Creating ensemble model...")
    ensemble = create_ensemble_model(trained_models)
    
    if ensemble:
        ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate all models
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    best_model = None
    best_score = -np.inf
    all_metrics = {}
    
    # Create model dictionary
    model_dict = dict(trained_models)
    if ensemble:
        model_dict['Ensemble'] = ensemble
    
    for name, model in model_dict.items():
        y_pred = model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test, y_pred)
        all_metrics[name] = metrics
        
        print(f"\n📈 {name}:")
        print(f"   R² Score: {metrics['R2']:.4f}")
        print(f"   RMSE: ${metrics['RMSE']:,.2f}")
        print(f"   MAE: ${metrics['MAE']:,.2f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = name
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model} (R²: {best_score:.4f})")
    print("="*60)
    
    # Save best model
    final_model = model_dict.get(best_model)
    
    # Save all files
    joblib.dump(final_model, 'models/best_model.pkl')
    joblib.dump(final_model, 'models/house_price_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    save_metrics(all_metrics)
    
    print("\n✅ Models saved successfully!")
    
    # Plot feature importance for tree-based models
    if best_model in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'CatBoost']:
        if best_model in trained_models:
            fig = plot_feature_importance(trained_models[best_model], feature_names)
            if fig:
                fig.savefig('models/feature_importance.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                print("📊 Feature importance plot saved!")
    
    return final_model, scaler, feature_names

if __name__ == "__main__":
    final_model, scaler, feature_names = main()
    print("\n🎉 Training completed successfully!")