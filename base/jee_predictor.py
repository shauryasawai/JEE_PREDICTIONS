import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class JEEPredictorV2:
    """
    Enhanced ML Model for predicting colleges based on JEE rank with updated data structure
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        self.feature_columns = []
        self.institutes_data = None
        
    def extract_institute_state(self, institute_name):
        """
        Extract the state from the institute name using keyword matching
        """
        if not isinstance(institute_name, str):
            return 'Unknown'

        state_mapping = {
            'Delhi': 'Delhi', 'New Delhi': 'Delhi',
            'Mumbai': 'Maharashtra', 'Bombay': 'Maharashtra', 'Pune': 'Maharashtra', 
            'Nagpur': 'Maharashtra', 'Nashik': 'Maharashtra',
            'Bangalore': 'Karnataka', 'Bengaluru': 'Karnataka', 'Mysore': 'Karnataka',
            'Hyderabad': 'Telangana', 'Warangal': 'Telangana',
            'Chennai': 'Tamil Nadu', 'Madras': 'Tamil Nadu', 'Tiruchirappalli': 'Tamil Nadu',
            'Trichy': 'Tamil Nadu', 'Coimbatore': 'Tamil Nadu',
            'Kolkata': 'West Bengal', 'Kharagpur': 'West Bengal', 'Durgapur': 'West Bengal',
            'Guwahati': 'Assam', 'Silchar': 'Assam',
            'Kanpur': 'Uttar Pradesh', 'Varanasi': 'Uttar Pradesh', 'Allahabad': 'Uttar Pradesh',
            'Prayagraj': 'Uttar Pradesh', 'Lucknow': 'Uttar Pradesh',
            'Roorkee': 'Uttarakhand', 'Dehradun': 'Uttarakhand',
            'Patna': 'Bihar', 'Gaya': 'Bihar',
            'Bhopal': 'Madhya Pradesh', 'Indore': 'Madhya Pradesh', 'Jabalpur': 'Madhya Pradesh',
            'Jaipur': 'Rajasthan', 'Jodhpur': 'Rajasthan', 'Kota': 'Rajasthan',
            'Ahmedabad': 'Gujarat', 'Surat': 'Gujarat', 'Gandhinagar': 'Gujarat', 'Rajkot': 'Gujarat',
            'Chandigarh': 'Chandigarh', 'Mohali': 'Punjab', 'Patiala': 'Punjab',
            'Rourkela': 'Odisha', 'Bhubaneswar': 'Odisha', 'Cuttack': 'Odisha',
            'Thiruvananthapuram': 'Kerala', 'Calicut': 'Kerala', 'Kozhikode': 'Kerala',
            'Thrissur': 'Kerala', 'Kochi': 'Kerala',
            'Mandi': 'Himachal Pradesh', 'Hamirpur': 'Himachal Pradesh',
            'Srinagar': 'Jammu and Kashmir', 'Jammu': 'Jammu and Kashmir',
            'Raipur': 'Chhattisgarh', 'Bhilai': 'Chhattisgarh',
            'Ranchi': 'Jharkhand', 'Dhanbad': 'Jharkhand', 'Jamshedpur': 'Jharkhand',
            'Goa': 'Goa', 'Panaji': 'Goa',
            'Manipur': 'Manipur', 'Imphal': 'Manipur',
            'Meghalaya': 'Meghalaya', 'Shillong': 'Meghalaya',
            'Mizoram': 'Mizoram', 'Aizawl': 'Mizoram',
            'Nagaland': 'Nagaland', 'Kohima': 'Nagaland', 'Dimapur': 'Nagaland',
            'Sikkim': 'Sikkim', 'Gangtok': 'Sikkim',
            'Tripura': 'Tripura', 'Agartala': 'Tripura',
            'Arunachal Pradesh': 'Arunachal Pradesh', 'Itanagar': 'Arunachal Pradesh',
            'Puducherry': 'Puducherry', 'Pondicherry': 'Puducherry',
            'Andaman': 'Andaman and Nicobar Islands',
            'Vijayawada': 'Andhra Pradesh', 'Visakhapatnam': 'Andhra Pradesh',
            'Tirupati': 'Andhra Pradesh', 'Anantapur': 'Andhra Pradesh'
        }
        institute_name = institute_name.lower()
        for keyword, state in state_mapping.items():
            if keyword.lower() in institute_name:
                return state
            
        return 'Unknown'

    
    def normalize_data(self, df):
        
        data = df.copy()

        data.columns = data.columns.str.strip()

        if 'Seat Type' in data.columns:
            data['Seat Type'] = data['Seat Type'].replace('GEN', 'OPEN')
            
            if 'Gender' in data.columns:
                data['Gender_Derived'] = data['Gender'].apply(
                    lambda x: 'Male' if x == 'Gender-Neutral' else 'Female'
                    )
                if 'Institute' in data.columns:
                    data['Institute'] = data['Institute'].fillna('Unknown')
                    data['institute_state'] = data['Institute'].apply(
                        self.extract_institute_state)
                    
                    column_mapping = {'Institute': 'institute',
                                      'Academic Program Name': 'program_name',
                                      'Quota': 'quota',
                                      'Seat Type': 'seat_type',
                                      'Gender': 'gender',
                                      'Opening Rank': 'opening_rank',
                                      'Closing Rank': 'closing_rank',
                                      'Closing Rank (CR)': 'closing_rank',
                                      'Opening Rank (OR)': 'opening_rank'
                                      }
                    data = data.rename(columns=column_mapping)
                    required_cols = ['opening_rank', 'closing_rank']
                    missing = [c for c in required_cols if c not in data.columns]
                    if missing:
                        print(f"\nAvailable columns: {data.columns.tolist()}")
                        raise ValueError(f"Missing required columns: {missing}")
                    return data
    
    def preprocess_data(self, df):
        """
        Preprocess the JEE admissions data with encoding
        """
        data = self.normalize_data(df)
        
        # Encode categorical variables
        categorical_cols = ['institute', 'program_name', 'quota', 'seat_type', 'gender']
        
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        data[col].astype(str)
                    )
                else:
                    # Handle unseen labels
                    known_labels = set(self.label_encoders[col].classes_)
                    data[f'{col}_encoded'] = data[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in known_labels else -1
                    )
        
        return data
    
    def prepare_training_data(self, df):
        """
        Prepare data for model training with optimized sampling
        """
        processed_df = self.preprocess_data(df)
        self.institutes_data = self.normalize_data(df)
        
        training_samples = []
        
        for idx, row in processed_df.iterrows():
            opening_rank = row['opening_rank']
            closing_rank = row['closing_rank']
            
            # Skip invalid data
            if pd.isna(opening_rank) or pd.isna(closing_rank):
                continue
            
            # Create training samples
            # 1. Definitely admitted (within opening to closing range)
            mid_rank = (opening_rank + closing_rank) // 2
            quarter_1 = opening_rank + (closing_rank - opening_rank) // 4
            quarter_3 = opening_rank + 3 * (closing_rank - opening_rank) // 4
            
            for rank in [opening_rank, quarter_1, mid_rank, quarter_3, closing_rank]:
                training_samples.append({
                    'rank': int(rank),
                    'seat_type_encoded': row['seat_type_encoded'],
                    'gender_encoded': row['gender_encoded'],
                    'quota_encoded': row['quota_encoded'],
                    'target_institute': row['institute_encoded'],
                    'target_program': row['program_name_encoded'],
                    'admitted': 1
                })
            
            # 2. Not admitted (beyond closing rank)
            if closing_rank < 400000:
                worse_rank_1 = min(int(closing_rank + 500), 500000)
                worse_rank_2 = min(int(closing_rank + 2000), 500000)
                
                for worse_rank in [worse_rank_1, worse_rank_2]:
                    training_samples.append({
                        'rank': worse_rank,
                        'seat_type_encoded': row['seat_type_encoded'],
                        'gender_encoded': row['gender_encoded'],
                        'quota_encoded': row['quota_encoded'],
                        'target_institute': row['institute_encoded'],
                        'target_program': row['program_name_encoded'],
                        'admitted': 0
                    })
        
        return pd.DataFrame(training_samples)
    
    def train_model(self, df):
        """
        Train the admission prediction model
        """
        print("Preparing training data...")
        training_data = self.prepare_training_data(df)
        
        self.feature_columns = ['rank', 'seat_type_encoded', 'gender_encoded',
                               'quota_encoded', 'target_institute', 'target_program']
        
        X = training_data[self.feature_columns]
        y = training_data['admitted']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
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
    
    def predict_colleges(self, rank, category_rank=None, gender='Male', 
                        home_state=None, preferences=None):
        """
        Predict colleges based on user information
        
        Parameters:
        -----------
        rank : int
            User's JEE rank (category-specific rank)
        category_rank : str
            Category seat type (OPEN, SC, ST, OBC-NCL, EWS, etc.)
        gender : str
            'Male' or 'Female'
        home_state : str
            User's home state (e.g., 'Assam', 'Delhi', 'Karnataka') for Home State quota
        preferences : dict
            Additional preferences like specific institutes or programs
        """
        if self.institutes_data is None:
            raise ValueError("Data not loaded. Please train or load the model first.")
        
        print(f"\nPredicting for:")
        print(f"  Rank: {rank}")
        print(f"  Category: {category_rank}")
        print(f"  Gender: {gender}")
        print(f"  Home State: {home_state}")
        
        predictions = []
        filtered_data = self.institutes_data.copy()
        
        # Filter by category (seat type)
        if category_rank:
            filtered_data = filtered_data[
                filtered_data['seat_type'] == category_rank
            ]
        
        # Filter by gender
        if gender == 'Male':
            filtered_data = filtered_data[
                filtered_data['gender'] == 'Gender-Neutral'
            ]
        elif gender == 'Female':
            # Female candidates can apply to both Gender-Neutral and Female-only seats
            filtered_data = filtered_data[
                filtered_data['gender'].isin(['Gender-Neutral', 
                                              'Female-only (including Supernumerary)',
                                              'Female-only'])
            ]
        
        # Apply quota filter based on home state
        if home_state:
            # For HS quota, only include institutes in the user's home state
            # Also include all AI (All India) quota seats (accessible to everyone)
            filtered_data = filtered_data[
                (filtered_data['quota'] == 'AI') | 
                ((filtered_data['quota'] == 'HS') & 
                 (filtered_data['institute_state'] == home_state))
            ]
        else:
            # Only All India quota if no home state specified
            filtered_data = filtered_data[filtered_data['quota'] == 'AI']
        
        # Apply additional preferences
        if preferences:
            if 'institute' in preferences and preferences['institute']:
                filtered_data = filtered_data[
                    filtered_data['institute'].str.contains(
                        preferences['institute'], case=False, na=False
                    )
                ]
            if 'program' in preferences and preferences['program']:
                filtered_data = filtered_data[
                    filtered_data['program_name'].str.contains(
                        preferences['program'], case=False, na=False
                    )
                ]
            if 'state' in preferences and preferences['state']:
                filtered_data = filtered_data[
                    filtered_data['institute_state'] == preferences['state']
                ]
        
        print(f"Found {len(filtered_data)} matching programs")
        
        # Show quota breakdown
        ai_count = len(filtered_data[filtered_data['quota'] == 'AI'])
        hs_count = len(filtered_data[filtered_data['quota'] == 'HS'])
        print(f"  - All India (AI) Quota: {ai_count} programs")
        print(f"  - Home State (HS) Quota: {hs_count} programs")
        
        # Generate predictions for each program
        for idx, row in filtered_data.iterrows():
            opening_rank = int(row['opening_rank'])
            closing_rank = int(row['closing_rank'])
            
            # FIXED LOGIC: Lower rank is better in JEE
            # User can get admission if their rank <= closing_rank
            
            if rank > closing_rank + 2000:
                # Rank is much worse than closing rank - skip
                continue
            elif rank > closing_rank + 500:
                # Slightly beyond closing rank
                status = "Very Low Chance"
                confidence = 20.0
                admission_prob = 15.0
            elif rank > closing_rank:
                # Just beyond closing rank
                status = "Low Chance"
                confidence = 40.0
                admission_prob = 35.0
            elif rank >= opening_rank and rank <= closing_rank:
                # Within admission range (between opening and closing)
                # Calculate position within range (0 = at opening rank, 1 = at closing rank)
                position = (rank - opening_rank) / max((closing_rank - opening_rank), 1)
                
                # Better rank (closer to opening) = higher confidence
                confidence = 90.0 - (position * 30.0)  # 90% at opening, 60% at closing
                admission_prob = confidence
                
                if position <= 0.33:
                    status = "High Chance"  # Top 33% of the range
                elif position <= 0.67:
                    status = "Good Chance"  # Middle 33% of the range
                else:
                    status = "Moderate Chance"  # Bottom 33% of the range
            else:
                # rank < opening_rank (better than opening rank!)
                status = "Excellent Chance"
                confidence = 98.0
                admission_prob = 98.0
            
            predictions.append({
                'institute': row['institute'],
                'program': row['program_name'],
                'quota': row['quota'],
                'seat_type': row['seat_type'],
                'gender': row['gender'],
                'institute_state': row['institute_state'],
                'opening_rank': opening_rank,
                'closing_rank': closing_rank,
                'your_rank': rank,
                'admission_probability': round(admission_prob, 2),
                'confidence': round(confidence, 2),
                'status': status
            })
        
        # Sort by confidence (highest first), then by closing rank (lower is better)
        predictions.sort(key=lambda x: (-x['confidence'], x['closing_rank']))
        
        print(f"Generated {len(predictions)} predictions")
        return predictions
    
    def save_model(self, model_path='jee_predictor_v2.pkl'):
        """Save the model and data"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'institutes_data': self.institutes_data
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='jee_predictor_v2.pkl'):
        """Load a pre-trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.institutes_data = model_data['institutes_data']
        print(f"Model loaded from {model_path}")


# Training and Testing Script
if __name__ == "__main__":
    import sys
    
    # Load your data
    try:
        df = pd.read_csv('jee_admissions_data.csv')
    except FileNotFoundError:
        print("Error: jee_admissions_data.csv not found!")
        sys.exit(1)
    
    print("Starting JEE Predictor V2 Training...")
    print(f"Total records: {len(df)}")
    
    # Initialize and train model
    predictor = JEEPredictorV2()
    accuracy = predictor.train_model(df)
    
    # Save the model
    predictor.save_model('jee_predictor_v2.pkl')
    
    # Test predictions with different scenarios
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    # Example 1: Male student with OPEN category from Assam
    print("\n--- Example 1: Male, OPEN category, Rank 5000, Home State: Assam ---")
    predictions = predictor.predict_colleges(
        rank=5000,
        category_rank='OPEN',
        gender='Male',
        home_state='Assam'
    )
    
    print(f"\nTop 10 Predictions:")
    for i, pred in enumerate(predictions[:10], 1):
        print(f"\n{i}. {pred['institute']} ({pred['institute_state']})")
        print(f"   Program: {pred['program']}")
        print(f"   Quota: {pred['quota']}, Seat Type: {pred['seat_type']}")
        print(f"   Status: {pred['status']} ({pred['confidence']}% confidence)")
        print(f"   Opening: {pred['opening_rank']}, Closing: {pred['closing_rank']}")
    
    # Example 2: Female student with OBC-NCL category from Karnataka
    print("\n--- Example 2: Female, OBC-NCL, Rank 15000, Home State: Karnataka ---")
    predictions = predictor.predict_colleges(
        rank=15000,
        category_rank='OBC-NCL',
        gender='Female',
        home_state='Karnataka'
    )
    
    print(f"\nTop 5 Predictions:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"\n{i}. {pred['institute']} ({pred['institute_state']})")
        print(f"   Program: {pred['program']}")
        print(f"   Quota: {pred['quota']}, Seat Type: {pred['seat_type']}")
        print(f"   Status: {pred['status']} ({pred['confidence']}% confidence)")
    
    # Example 3: Student from Delhi looking for Computer Science
    print("\n--- Example 3: Male, OPEN, Rank 2000, Home State: Delhi, CS programs ---")
    predictions = predictor.predict_colleges(
        rank=2000,
        category_rank='OPEN',
        gender='Male',
        home_state='Delhi',
        preferences={'program': 'Computer Science'}
    )
    
    print(f"\nTop 5 CS Predictions:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"\n{i}. {pred['institute']} ({pred['institute_state']})")
        print(f"   Program: {pred['program']}")
        print(f"   Quota: {pred['quota']} - {pred['status']} ({pred['confidence']}% confidence)")
    
    # Example 4: Student looking only in Tamil Nadu
    print("\n--- Example 4: Male, SC, Rank 8000, Home State: Tamil Nadu, TN colleges only ---")
    predictions = predictor.predict_colleges(
        rank=8000,
        category_rank='SC',
        gender='Male',
        home_state='Tamil Nadu',
        preferences={'state': 'Tamil Nadu'}
    )
    
    print(f"\nTop 5 Tamil Nadu Predictions:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"\n{i}. {pred['institute']} ({pred['institute_state']})")
        print(f"   Program: {pred['program']}")
        print(f"   Quota: {pred['quota']} - {pred['status']} ({pred['confidence']}% confidence)")