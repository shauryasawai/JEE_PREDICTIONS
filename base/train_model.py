import pandas as pd
from jee_predictor import JEEPredictor
import os

# Load your data
data_path = os.path.join('data', 'data.csv')
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} records")
print(f"Institutes: {df['institute_short'].nunique()}")
print(f"Programs: {df['program_name'].nunique()}")

# Initialize and train
predictor = JEEPredictor()
print("\nTraining model...")
accuracy = predictor.train_model(df)

# Save model
model_path = os.path.join('base', 'jee_predictor_model.pkl')
predictor.save_model(model_path)

print(f"\n✅ Model training complete!")
print(f"Model saved to: {model_path}")
print(f"Accuracy: {accuracy:.4f}")

# Test prediction
print("\n" + "="*60)
print("TEST PREDICTION")
print("="*60)
test_rank = 1500
test_category = "GEN"
predictions = predictor.predict_colleges(test_rank, test_category)
print(f"\nRank: {test_rank}, Category: {test_category}")
print(f"Total predictions: {len(predictions)}")
print("\nTop 5 predictions:")
for i, pred in enumerate(predictions[:5], 1):
    print(f"{i}. {pred['institute']} - {pred['program']}")
    print(f"   Status: {pred['status']} ({pred['confidence']}%)")