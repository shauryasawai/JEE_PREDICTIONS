import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class JEEPredictor:
    """
    Optimized ML Model for predicting colleges based on JEE rank
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        self.feature_columns = []
        self.institutes_data = None
        
    def preprocess_data(self, df):
        """
        Preprocess the JEE admissions data
        """
        data = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['institute_type', 'quota', 'pool', 'institute_short', 
                          'program_name', 'degree_short', 'category']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col].astype(str))
        
        return data
    
    def prepare_training_data(self, df):
        """
        Prepare data for model training - OPTIMIZED VERSION
        """
        processed_df = self.preprocess_data(df)
        self.institutes_data = df.copy()
        
        training_samples = []
        
        # Reduced sampling for faster training
        for idx, row in processed_df.iterrows():
            opening_rank = row['opening_rank']
            closing_rank = row['closing_rank']
            
            # Create fewer samples per program (3 instead of 7)
            # Within range - admitted
            mid_rank = (opening_rank + closing_rank) // 2
            for rank in [opening_rank, mid_rank, closing_rank]:
                training_samples.append({
                    'rank': rank,
                    'category_encoded': row['category_encoded'],
                    'institute_type_encoded': row['institute_type_encoded'],
                    'quota_encoded': row['quota_encoded'],
                    'pool_encoded': row['pool_encoded'],
                    'target_institute': row['institute_short_encoded'],
                    'target_program': row['program_name_encoded'],
                    'admitted': 1
                })
            
            # Outside range - not admitted
            if closing_rank < 400000:
                worse_rank = min(closing_rank + 1000, 500000)
                training_samples.append({
                    'rank': worse_rank,
                    'category_encoded': row['category_encoded'],
                    'institute_type_encoded': row['institute_type_encoded'],
                    'quota_encoded': row['quota_encoded'],
                    'pool_encoded': row['pool_encoded'],
                    'target_institute': row['institute_short_encoded'],
                    'target_program': row['program_name_encoded'],
                    'admitted': 0
                })
        
        return pd.DataFrame(training_samples)
    
    def train_model(self, df):
        """
        Train the admission prediction model - OPTIMIZED
        """
        print("Preparing training data...")
        training_data = self.prepare_training_data(df)
        
        self.feature_columns = ['rank', 'category_encoded', 'institute_type_encoded', 
                               'quota_encoded', 'pool_encoded', 'target_institute', 
                               'target_program']
        
        X = training_data[self.feature_columns]
        y = training_data['admitted']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Faster Random Forest with fewer trees
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=15,      # Reduced from 20
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict_colleges(self, rank, category, preferences=None):
        """
        OPTIMIZED: Predict colleges based on rank - NO ML, PURE LOGIC
        This is much faster and more reliable for rank-based admission
        """
        if self.institutes_data is None:
            raise ValueError("Data not loaded. Please train or load the model first.")
        
        print(f"Predicting for rank={rank}, category={category}")
        
        predictions = []
        filtered_data = self.institutes_data.copy()
        
        # Apply preferences filter
        if preferences:
            for key, value in preferences.items():
                if key in filtered_data.columns and value:
                    filtered_data = filtered_data[filtered_data[key] == value]
        
        # Filter by category
        filtered_data = filtered_data[filtered_data['category'] == category]
        
        print(f"Found {len(filtered_data)} matching programs")
        
        # Process each college-branch combination
        for idx, row in filtered_data.iterrows():
            opening_rank = int(row['opening_rank'])
            closing_rank = int(row['closing_rank'])
            
            # Determine admission status based on rank position
            if rank <= opening_rank:
                status = "High Chance"
                confidence = 95.0
                admission_prob = 95.0
            elif rank <= closing_rank:
                # Within admission range
                position = (rank - opening_rank) / (closing_rank - opening_rank)
                confidence = 90.0 - (position * 30.0)  # 90% to 60%
                admission_prob = confidence
                
                if confidence >= 75:
                    status = "Good Chance"
                else:
                    status = "Moderate Chance"
            else:
                # Beyond closing rank
                excess = rank - closing_rank
                if excess < 500:
                    status = "Low Chance"
                    confidence = 40.0
                    admission_prob = 35.0
                elif excess < 2000:
                    status = "Very Low Chance"
                    confidence = 20.0
                    admission_prob = 15.0
                else:
                    # Skip predictions with very low probability
                    continue
            
            predictions.append({
                'institute': row['institute_short'],
                'program': row['program_name'],
                'degree': row['degree_short'],
                'category': category,
                'quota': row['quota'],
                'opening_rank': opening_rank,
                'closing_rank': closing_rank,
                'your_rank': rank,
                'admission_probability': round(admission_prob, 2),
                'confidence': round(confidence, 2),
                'status': status,
                'year': row['year'],
                'round': row['round_no']
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Generated {len(predictions)} predictions")
        return predictions
    
    def save_model(self, model_path='jee_predictor_model.pkl'):
        """
        Save the model and data
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'institutes_data': self.institutes_data
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='jee_predictor_model.pkl'):
        """
        Load a pre-trained model
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.institutes_data = model_data['institutes_data']
        print(f"Model loaded from {model_path}")


# Training script
if __name__ == "__main__":
    import sys
    
    # Load your data
    try:
        df = pd.read_csv('jee_admissions_data.csv')
    except FileNotFoundError:
        print("Error: jee_admissions_data.csv not found!")
        sys.exit(1)
    
    print("Starting JEE Predictor Training...")
    print(f"Total records: {len(df)}")
    
    # Initialize and train model
    predictor = JEEPredictor()
    accuracy = predictor.train_model(df)
    
    # Save the model
    predictor.save_model('jee_predictor_model.pkl')
    
    # Test prediction
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    test_rank = 1500
    test_category = "GEN"
    
    print(f"\nPredicting colleges for Rank: {test_rank}, Category: {test_category}")
    predictions = predictor.predict_colleges(
        rank=test_rank,
        category=test_category
    )
    
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"\nTop 10 Predictions:")
    for i, pred in enumerate(predictions[:10], 1):
        print(f"\n{i}. {pred['institute']} - {pred['program']}")
        print(f"   Status: {pred['status']} ({pred['confidence']}% confidence)")
        print(f"   Opening: {pred['opening_rank']}, Closing: {pred['closing_rank']}")