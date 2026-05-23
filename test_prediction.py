import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

print("🏠 House Price Predictor")
print("="*40)

# Example house details
example_house = {
    'area': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'age': 5,
    'location_score': 7.5,
    'garage_spaces': 1,
    'has_basement': 1,
    'has_pool': 0
}

# Create feature array (must match training features)
# For Lasso, we need all engineered features
features = np.array([[
    example_house['area'],
    example_house['bedrooms'],
    example_house['bathrooms'],
    example_house['age'],
    example_house['location_score'],
    example_house['garage_spaces'],
    example_house['has_basement'],
    example_house['has_pool'],
    example_house['area'] / example_house['bedrooms'],  # area_per_bedroom
    example_house['area'] / example_house['bathrooms'], # area_per_bathroom
    example_house['bedrooms'] / example_house['bathrooms'], # bed_bath_ratio
    example_house['location_score'] / (example_house['age'] + 1), # location_age_interaction
    (example_house['garage_spaces'] * 0.3 + 
     example_house['has_basement'] * 0.3 + 
     example_house['has_pool'] * 0.4), # luxury_score
    0, 0, 0,  # age category dummies (adjust based on your features)
    0, 0      # location category dummies
]])

# Scale features
features_scaled = scaler.transform(features)

# Predict
predicted_price = model.predict(features_scaled)[0]

print(f"\n📊 House Details:")
print(f"   Area: {example_house['area']} sq ft")
print(f"   Bedrooms: {example_house['bedrooms']}")
print(f"   Bathrooms: {example_house['bathrooms']}")
print(f"   Age: {example_house['age']} years")
print(f"   Location Score: {example_house['location_score']}/10")
print(f"   Garage: {example_house['garage_spaces']} spaces")
print(f"   Basement: {'Yes' if example_house['has_basement'] else 'No'}")
print(f"   Pool: {'Yes' if example_house['has_pool'] else 'No'}")

print(f"\n💰 PREDICTED PRICE: ${predicted_price:,.2f}")
print(f"≈ ₹{predicted_price * 83:,.2f} INR")