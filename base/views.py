from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
import warnings
import traceback
import joblib
import requests
from io import BytesIO
from functools import lru_cache
from .jee_predictor import JEEPredictorV2
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Initialize predictor globally
predictor = None

# Model URL from GitHub Release
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/shauryasawai/JEE_PREDICTIONS/releases/download/v1.0.0-JEE_prediction/jee_predictor_v2.pkl"
)

# Local storage path
LOCAL_MODEL_DIR = Path("models")
LOCAL_MODEL_PATH = LOCAL_MODEL_DIR / "jee_predictor_v2.pkl"

def ensure_model_directory():
    """Create models directory if it doesn't exist"""
    LOCAL_MODEL_DIR.mkdir(exist_ok=True)

@lru_cache(maxsize=1)
def download_and_load_model():
    """
    Download model from GitHub release, save locally, and cache it in memory
    """
    try:
        ensure_model_directory()
        
        # Check if model exists locally
        if LOCAL_MODEL_PATH.exists():
            print(f"Loading model from local storage: {LOCAL_MODEL_PATH}")
            model_data = joblib.load(LOCAL_MODEL_PATH)
            print("Model loaded successfully from local storage!")
            return model_data
        
        # Download model if not available locally
        print(f"Model not found locally. Downloading from: {MODEL_URL}")
        response = requests.get(MODEL_URL, timeout=60)
        response.raise_for_status()
        
        print("Model downloaded, saving to local storage...")
        # Save to local file
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print(f"Model saved to: {LOCAL_MODEL_PATH}")
        
        # Load from local file
        print("Loading model into memory...")
        model_data = joblib.load(LOCAL_MODEL_PATH)
        print("Model loaded successfully!")
        return model_data
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to load model: {str(e)}")

def load_predictor():
    """Load the predictor model from local storage or external URL"""
    global predictor
    if predictor is None:
        try:
            print("Initializing JEE Predictor V2...")
            predictor = JEEPredictorV2()
            
            # Download and load model (or load from local if available)
            model_data = download_and_load_model()
            
            # Load the model data into predictor
            predictor.model = model_data.get('model')
            predictor.label_encoders = model_data.get('label_encoders', {})
            predictor.feature_columns = model_data.get('feature_columns', [])
            predictor.institutes_data = model_data.get('institutes_data')
            
            if predictor.institutes_data is None:
                raise Exception("Model data is corrupted or incomplete")
            
            print(f"Predictor loaded with {len(predictor.institutes_data)} records")
            return True
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            traceback.print_exc()
            return False
    return True

def clear_local_model():
    """
    Utility function to delete local model file 
    (useful for forcing a re-download)
    """
    try:
        if LOCAL_MODEL_PATH.exists():
            LOCAL_MODEL_PATH.unlink()
            print(f"Deleted local model: {LOCAL_MODEL_PATH}")
            # Clear the cache
            download_and_load_model.cache_clear()
            return True
    except Exception as e:
        print(f"Error deleting local model: {e}")
        return False

# Load model on startup
load_predictor()


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """
    Handle prediction requests
    POST /api/predict/
    
    Request body (JSON):
    {
        "rank": 10000,
        "category": "OPEN",
        "gender": "Male",
        "home_state": "Delhi",  # Optional
        "preferences": {  # Optional
            "institute": "IIT Delhi",
            "program": "Computer Science",
            "state": "Delhi"
        }
    }
    """
    try:
        data = json.loads(request.body)
        
        # Extract parameters
        rank = int(data.get('rank'))
        category = data.get('category', 'OPEN')
        gender = data.get('gender', 'Male')
        home_state = data.get('home_state', None)
        preferences = data.get('preferences', {})
        
        # Handle empty home_state
        if home_state in ['none', '', None]:
            home_state = None
        
        # Validate inputs
        if rank <= 0 or rank > 500000:
            return JsonResponse({
                'success': False,
                'error': 'Invalid rank. Please enter a rank between 1 and 500000'
            }, status=400)
        
        if not predictor or predictor.institutes_data is None:
            return JsonResponse({
                'success': False,
                'error': 'Model not loaded'
            }, status=500)
        
        print(f"\nPREDICTION - Rank: {rank}, Category: {category}, Gender: {gender}, State: {home_state}")
        
        # DEBUG: Check what categories exist in the data
        df = predictor.institutes_data
        print(f"DEBUG: Available seat_types in data: {df['seat_type'].unique().tolist() if 'seat_type' in df.columns else 'N/A'}")
        print(f"DEBUG: Total records in dataset: {len(df)}")
        print(f"DEBUG: Sample of seat_type values: {df['seat_type'].value_counts().head() if 'seat_type' in df.columns else 'N/A'}")
        
        # Get predictions
        predictions = predictor.predict_colleges(
            rank=rank,
            category_rank=category,
            gender=gender,
            home_state=home_state,
            preferences=preferences
        )
        
        print(f"Results: {len(predictions)} total predictions")
        
        # If no predictions found, return empty string
        if len(predictions) == 0:
            return JsonResponse({
                'success': True,
                'rank': rank,
                'category': category,
                'gender': gender,
                'home_state': home_state,
                'total_predictions': 0,
                'predictions': {
                    'high_chance': [],
                    'good_chance': [],
                    'moderate_chance': [],
                    'low_chance': [],
                    'all': []
                },
                'message': ''
            })
        
        # Categorize predictions
        categorized = {
            'high_chance': [p for p in predictions if p['status'] == 'High Chance'],
            'good_chance': [p for p in predictions if p['status'] == 'Good Chance'],
            'moderate_chance': [p for p in predictions if p['status'] == 'Moderate Chance'],
            'low_chance': [p for p in predictions if p['status'] in ['Low Chance', 'Very Low Chance']],
        }
        
        return JsonResponse({
            'success': True,
            'rank': rank,
            'category': category,
            'gender': gender,
            'home_state': home_state,
            'total_predictions': len(predictions),
            'predictions': {
                'high_chance': categorized['high_chance'][:20],
                'good_chance': categorized['good_chance'][:20],
                'moderate_chance': categorized['moderate_chance'][:20],
                'low_chance': categorized['low_chance'][:10],
                'all': predictions[:100]
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON format'
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }, status=400)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_options(request):
    """
    Get available options for the prediction form
    GET /api/options/
    """
    if not predictor or predictor.institutes_data is None:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded'
        }, status=500)
    
    indian_states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
        'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
        'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
    ]
    
    return JsonResponse({
        'success': True,
        'categories': ['OPEN', 'OBC-NCL', 'SC', 'ST', 'EWS', 'OPEN (PwD)', 
                      'OBC-NCL (PwD)', 'SC (PwD)', 'ST (PwD)', 'EWS (PwD)'],
        'genders': ['Male', 'Female'],
        'states': sorted(indian_states)
    })


@require_http_methods(["GET"])
def health(request):
    """
    Health check endpoint
    GET /api/health/
    """
    return JsonResponse({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'data_available': predictor.institutes_data is not None if predictor else False
    })


@require_http_methods(["GET"])
def debug_data(request):
    """
    Debug endpoint to inspect the dataset
    GET /api/debug/
    """
    if not predictor or predictor.institutes_data is None:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded'
        }, status=500)
    
    df = predictor.institutes_data
    
    debug_info = {
        'success': True,
        'total_records': len(df),
        'columns': df.columns.tolist(),
        'sample_data': df.head(5).to_dict('records'),
        'unique_values': {}
    }
    
    # Get unique values for important columns
    important_cols = ['seat_type', 'gender', 'quota', 'institute_state']
    for col in important_cols:
        if col in df.columns:
            debug_info['unique_values'][col] = {
                'count': df[col].nunique(),
                'values': df[col].unique().tolist()[:20]
            }
    
    # Get value counts for seat_type
    if 'seat_type' in df.columns:
        debug_info['seat_type_distribution'] = df['seat_type'].value_counts().head(10).to_dict()
    
    return JsonResponse(debug_info)