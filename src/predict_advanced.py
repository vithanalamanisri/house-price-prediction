"""
Advanced prediction module for house price prediction
"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class HousePricePredictor:
    """Main predictor class for house prices"""
    
    def __init__(self):
        """Initialize the predictor with trained models"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            model_path = PROJECT_ROOT / 'models' / 'best_model.pkl'
            scaler_path = PROJECT_ROOT / 'models' / 'scaler.pkl'
            features_path = PROJECT_ROOT / 'models' / 'feature_names.pkl'
            
            # Try to load best_model first, fallback to house_price_model
            if model_path.exists():
                self.model = joblib.load(model_path)
                print("✅ Loaded best_model.pkl")
            else:
                fallback_path = PROJECT_ROOT / 'models' / 'house_price_model.pkl'
                if fallback_path.exists():
                    self.model = joblib.load(fallback_path)
                    print("✅ Loaded house_price_model.pkl")
                else:
                    raise FileNotFoundError("No model found. Please train first.")
            
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            print("✅ Models and preprocessors loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_single(self, features_dict):
        """
        Predict price for a single house
        
        Args:
            features_dict: Dictionary with house features
            
        Returns:
            float: Predicted price
        """
        try:
            # Create dataframe with all required features
            features_df = pd.DataFrame([features_dict])
            
            # Ensure all required features are present
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Reorder columns to match training
            features_df = features_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def predict_batch(self, houses_df):
        """
        Predict prices for multiple houses
        
        Args:
            houses_df: DataFrame with house features
            
        Returns:
            np.array: Array of predicted prices
        """
        try:
            # Ensure all required features are present
            for col in self.feature_names:
                if col not in houses_df.columns:
                    houses_df[col] = 0
            
            # Reorder columns
            houses_df = houses_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(houses_df)
            
            # Predict
            predictions = self.model.predict(features_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return None
    
    def get_price_range(self, features_dict, confidence=0.95):
        """
        Get price range with confidence interval
        
        Args:
            features_dict: Dictionary with house features
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            tuple: (lower_bound, upper_bound, predicted_price)
        """
        predicted = self.predict_single(features_dict)
        
        if predicted is None:
            return None, None, None
        
        # Simple confidence interval based on model's typical error
        # In production, you'd use more sophisticated methods
        error_margin = predicted * 0.1  # 10% margin
        
        lower_bound = predicted - error_margin
        upper_bound = predicted + error_margin
        
        return lower_bound, upper_bound, predicted


def create_features_dict(area, bedrooms, bathrooms, age, location_score, 
                        garage_spaces=1, has_basement=0, has_pool=0, 
                        recently_renovated=0):
    """
    Helper function to create features dictionary from basic inputs
    
    Args:
        area: Square footage
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        age: Property age in years
        location_score: Location score (1-10)
        garage_spaces: Number of garage spaces (0-3)
        has_basement: 1 if has basement, else 0
        has_pool: 1 if has pool, else 0
        recently_renovated: 1 if renovated recently, else 0
        
    Returns:
        dict: Features dictionary
    """
    # Calculate derived features
    area_per_bedroom = area / max(bedrooms, 1)
    area_per_bathroom = area / max(bathrooms, 1)
    bed_bath_ratio = bedrooms / max(bathrooms, 1)
    location_age_interaction = location_score / (age + 1)
    luxury_score = (garage_spaces * 0.3 + has_basement * 0.3 + has_pool * 0.4)
    
    # Create age category dummies (0 for default)
    age_new = 1 if age <= 5 else 0
    age_recent = 1 if 5 < age <= 15 else 0
    age_moderate = 1 if 15 < age <= 30 else 0
    
    # Create location category dummies
    loc_high = 1 if location_score > 7 else 0
    loc_medium = 1 if 3 < location_score <= 7 else 0
    
    features = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'garage_spaces': garage_spaces,
        'has_basement': has_basement,
        'has_pool': has_pool,
        'recently_renovated': recently_renovated,
        'area_per_bedroom': area_per_bedroom,
        'area_per_bathroom': area_per_bathroom,
        'bed_bath_ratio': bed_bath_ratio,
        'location_age_interaction': location_age_interaction,
        'luxury_score': luxury_score,
        'age_category_New': age_new,
        'age_category_Recent': age_recent,
        'age_category_Moderate': age_moderate,
        'location_category_High': loc_high,
        'location_category_Medium': loc_medium
    }
    
    return features


def interactive_prediction():
    """Interactive command-line prediction interface"""
    print("\n" + "="*60)
    print("🏠 HOUSE PRICE PREDICTION SYSTEM")
    print("="*60)
    
    predictor = HousePricePredictor()
    
    if not predictor.model:
        print("\n❌ Please train the model first using: python src/train_advanced.py")
        return
    
    while True:
        print("\n📝 Enter house details:")
        print("-"*40)
        
        try:
            area = float(input("Area (sq ft) [500-5000]: "))
            bedrooms = int(input("Number of bedrooms [1-6]: "))
            bathrooms = float(input("Number of bathrooms [1-5]: "))
            age = int(input("Age of house (years) [0-60]: "))
            location_score = float(input("Location score (1-10): "))
            garage_spaces = int(input("Garage spaces [0-3]: "))
            
            has_basement = input("Has basement? (y/n): ").lower() == 'y'
            has_pool = input("Has swimming pool? (y/n): ").lower() == 'y'
            renovated = input("Recently renovated? (y/n): ").lower() == 'y'
            
            # Create features dictionary
            features = create_features_dict(
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                age=age,
                location_score=location_score,
                garage_spaces=garage_spaces,
                has_basement=1 if has_basement else 0,
                has_pool=1 if has_pool else 0,
                recently_renovated=1 if renovated else 0
            )
            
            # Get prediction with range
            lower, upper, price = predictor.get_price_range(features)
            
            if price:
                print("\n" + "="*40)
                print("💰 PREDICTION RESULTS")
                print("="*40)
                print(f"Predicted Price: ${price:,.2f}")
                print(f"Price Range (95% CI): ${lower:,.2f} - ${upper:,.2f}")
                print(f"≈ ₹{price*83:,.2f} (INR)")
                
                # Additional insights
                if price < 200000:
                    print("\n💡 Insight: Budget-friendly property")
                elif price < 400000:
                    print("\n💡 Insight: Mid-range property")
                else:
                    print("\n💡 Insight: Premium/Luxury property")
                print("="*40)
            
            another = input("\nPredict another house? (y/n): ").lower()
            if another != 'y':
                break
                
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def batch_prediction_from_csv(csv_path):
    """
    Predict prices for houses in CSV file
    
    Args:
        csv_path: Path to CSV file with house features
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} houses from {csv_path}")
        
        predictor = HousePricePredictor()
        
        if not predictor.model:
            print("❌ Model not loaded. Please train first.")
            return
        
        predictions = predictor.predict_batch(df)
        
        if predictions is not None:
            df['predicted_price'] = predictions
            df['price_range_low'] = predictions * 0.9
            df['price_range_high'] = predictions * 1.1
            
            # Save results
            output_path = csv_path.replace('.csv', '_predictions.csv')
            df.to_csv(output_path, index=False)
            print(f"\n✅ Predictions saved to {output_path}")
            
            # Display summary
            print("\n📊 Prediction Summary:")
            print(f"Min Price: ${df['predicted_price'].min():,.2f}")
            print(f"Max Price: ${df['predicted_price'].max():,.2f}")
            print(f"Avg Price: ${df['predicted_price'].mean():,.2f}")
            
            return df
    
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'batch' and len(sys.argv) > 2:
            batch_prediction_from_csv(sys.argv[2])
        else:
            print("Usage: python predict_advanced.py batch <csv_file_path>")
    else:
        interactive_prediction()