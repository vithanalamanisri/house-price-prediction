import joblib
import numpy as np
import pandas as pd

def load_model():
    """Load trained model and preprocessors"""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

def predict_price(model, scaler, features_dict):
    """Predict price from features dictionary"""
    # Create feature array (you need to match your exact feature set)
    # For now, let's use basic features
    area = features_dict['area']
    bedrooms = features_dict['bedrooms']
    bathrooms = features_dict['bathrooms']
    age = features_dict['age']
    location = features_dict['location_score']
    garage = features_dict['garage_spaces']
    basement = features_dict['has_basement']
    pool = features_dict['has_pool']
    
    # Calculate engineered features
    area_per_bedroom = area / max(bedrooms, 1)
    area_per_bathroom = area / max(bathrooms, 1)
    bed_bath_ratio = bedrooms / max(bathrooms, 1)
    location_age_interaction = location / (age + 1)
    luxury_score = (garage * 0.3 + basement * 0.3 + pool * 0.4)
    
    # Age category dummies (based on age)
    age_new = 1 if age <= 5 else 0
    age_recent = 1 if 5 < age <= 15 else 0
    age_moderate = 1 if 15 < age <= 30 else 0
    
    # Location category dummies
    loc_high = 1 if location > 7 else 0
    loc_medium = 1 if 3 < location <= 7 else 0
    
    # Create feature array with all features
    features = np.array([[
        area, bedrooms, bathrooms, age, location,
        garage, basement, pool,
        area_per_bedroom, area_per_bathroom, bed_bath_ratio,
        location_age_interaction, luxury_score,
        age_new, age_recent, age_moderate,
        loc_high, loc_medium
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]
    
    return price

def main():
    print("\n" + "="*60)
    print("🏠 INTERACTIVE HOUSE PRICE PREDICTOR")
    print("="*60)
    
    try:
        model, scaler, feature_names = load_model()
        print("✅ Model loaded successfully!\n")
        
        while True:
            print("\n📝 Enter House Details:")
            print("-"*40)
            
            try:
                area = float(input("Area (sq ft) [500-5000]: "))
                bedrooms = int(input("Number of bedrooms [1-6]: "))
                bathrooms = float(input("Number of bathrooms [1-5]: "))
                age = int(input("Age of house (years) [0-60]: "))
                location = float(input("Location score [1-10]: "))
                garage = int(input("Garage spaces [0-3]: "))
                
                basement = input("Has basement? (y/n): ").lower() == 'y'
                pool = input("Has swimming pool? (y/n): ").lower() == 'y'
                
                features = {
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'age': age,
                    'location_score': location,
                    'garage_spaces': garage,
                    'has_basement': int(basement),
                    'has_pool': int(pool)
                }
                
                price = predict_price(model, scaler, features)
                
                print("\n" + "="*40)
                print("💰 PREDICTION RESULTS")
                print("="*40)
                print(f"Predicted Price: ${price:,.2f}")
                print(f"≈ ₹{price * 83:,.2f} (INR)")
                
                # Price range with 10% margin
                print(f"\nPrice Range: ${price*0.9:,.2f} - ${price*1.1:,.2f}")
                
                # Investment advice
                if price < 200000:
                    print("\n💡 Budget-friendly property - Good for first-time buyers")
                elif price < 400000:
                    print("\n💡 Mid-range property - Good value for money")
                else:
                    print("\n💡 Premium property - Luxury investment")
                    
                print("="*40)
                
                another = input("\nPredict another house? (y/n): ").lower()
                if another != 'y':
                    break
                    
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    main()