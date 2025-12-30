from django.shortcuts import render
from django.http import JsonResponse
import json
import os
import warnings
import traceback
import joblib
import requests
from io import BytesIO
from functools import lru_cache
from .jee_predictor import JEEPredictorV2

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Initialize predictor globally
predictor = None

# Model URL from GitHub Release
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/shauryasawai/JEE_PREDICTIONS/releases/download/v1.0.0-JEE_prediction/jee_predictor_v2.pkl"
)

MODEL_PATH = "jee_predictor_v2.pkl"

def load_model():
    """Load model from local file or download if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading ML model...")
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    return joblib.load(MODEL_PATH)

@lru_cache(maxsize=1)
def download_and_load_model():
    """
    Download model from GitHub release and cache it in memory
    This function is called once and cached for subsequent requests
    """
    try:
        print(f"Downloading model from: {MODEL_URL}")
        response = requests.get(MODEL_URL, timeout=60)
        response.raise_for_status()
        
        print("Model downloaded, loading into memory...")
        model_data = joblib.load(BytesIO(response.content))
        print("Model loaded successfully!")
        return model_data
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to download model from {MODEL_URL}: {str(e)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to load model: {str(e)}")

def load_predictor():
    """Load the predictor model from external URL"""
    global predictor
    if predictor is None:
        try:
            print("Initializing JEE Predictor V2...")
            predictor = JEEPredictorV2()
            
            # Download and load model from GitHub
            model_data = download_and_load_model()
            
            # Load the model data into predictor
            predictor.model = model_data.get('model')
            predictor.label_encoders = model_data.get('label_encoders', {})
            predictor.feature_columns = model_data.get('feature_columns', [])
            predictor.institutes_data = model_data.get('institutes_data')
            
            if predictor.institutes_data is None:
                raise Exception("Model data is corrupted or incomplete")
            
            print(f"Predictor loaded with {len(predictor.institutes_data)} records")
            print(f"Available states: {predictor.institutes_data['institute_state'].nunique()}")
            return True
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            traceback.print_exc()
            return False
    return True

# Load model on startup
load_predictor()


def index(request):
    """
    Home page with prediction form
    """
    # Check if model is loaded
    if not predictor or predictor.institutes_data is None:
        return render(request, 'error.html', {
            'error': 'Model not loaded. Please contact administrator.',
            'details': f'Model URL: {MODEL_URL}'
        })
    
    # Indian states list
    indian_states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
        'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
        'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
    ]
    
    context = {
        'categories': ['OPEN', 'OBC-NCL', 'SC', 'ST', 'EWS', 'OPEN (PwD)', 'OBC-NCL (PwD)', 'SC (PwD)', 'ST (PwD)', 'EWS (PwD)'],
        'genders': ['Male', 'Female'],
        'states': sorted(indian_states),
        'institutes': get_unique_values('institute'),
        'programs': get_unique_values('program_name'),
        'available_states': sorted(predictor.institutes_data['institute_state'].unique().tolist()),
    }
    return render(request, 'index.html', context)


def predict(request):
    """
    Handle prediction requests with updated parameters
    """
    if request.method == 'POST':
        try:
            # Debug: Print received data
            print("\n" + "="*80)
            print("PREDICTION REQUEST RECEIVED (V2)")
            print("="*80)
            
            # Get form data
            rank = int(request.POST.get('rank'))
            category = request.POST.get('category', 'OPEN')
            gender = request.POST.get('gender', 'Male')
            home_state = request.POST.get('home_state', None)
            
            # Handle empty home_state
            if home_state == 'none' or home_state == '':
                home_state = None
            
            print(f"Rank: {rank}")
            print(f"Category: {category}")
            print(f"Gender: {gender}")
            print(f"Home State: {home_state}")
            
            # Validate inputs
            if rank <= 0 or rank > 500000:
                return render(request, 'error.html', {
                    'error': 'Invalid rank',
                    'details': 'Please enter a valid rank between 1 and 500000'
                })
            
            if not predictor or predictor.institutes_data is None:
                return render(request, 'error.html', {
                    'error': 'Model not loaded',
                    'details': 'Please contact administrator'
                })
            
            # Optional preferences
            preferences = {}
            institute_filter = request.POST.get('institute_filter')
            program_filter = request.POST.get('program_filter')
            state_filter = request.POST.get('state_filter')
            
            if institute_filter and institute_filter != 'all':
                preferences['institute'] = institute_filter
            if program_filter and program_filter != 'all':
                preferences['program'] = program_filter
            if state_filter and state_filter != 'all':
                preferences['state'] = state_filter
            
            print(f"Preferences: {preferences}")
            print("Starting prediction...")
            
            # Get predictions using V2 predictor
            predictions = predictor.predict_colleges(
                rank=rank,
                category_rank=category,
                gender=gender,
                home_state=home_state,
                preferences=preferences
            )
            
            print(f"Total predictions: {len(predictions)}")
            
            # Categorize predictions
            high_chance = [p for p in predictions if p['status'] == 'High Chance']
            good_chance = [p for p in predictions if p['status'] == 'Good Chance']
            moderate_chance = [p for p in predictions if p['status'] == 'Moderate Chance']
            low_chance = [p for p in predictions if p['status'] in ['Low Chance', 'Very Low Chance']]
            
            # Separate by quota
            ai_quota = [p for p in predictions if p['quota'] == 'AI']
            hs_quota = [p for p in predictions if p['quota'] == 'HS']
            
            print(f"High: {len(high_chance)}, Good: {len(good_chance)}, Moderate: {len(moderate_chance)}, Low: {len(low_chance)}")
            print(f"AI Quota: {len(ai_quota)}, HS Quota: {len(hs_quota)}")
            print("="*80 + "\n")
            
            context = {
                'rank': rank,
                'category': category,
                'gender': gender,
                'home_state': home_state,
                'total_predictions': len(predictions),
                'high_chance': high_chance[:20],
                'good_chance': good_chance[:20],
                'moderate_chance': moderate_chance[:20],
                'low_chance': low_chance[:10],
                'ai_quota': ai_quota[:30],
                'hs_quota': hs_quota[:30],
                'all_predictions': predictions[:100],
            }
            
            return render(request, 'results.html', context)
            
        except ValueError as e:
            print(f"ValueError: {e}")
            traceback.print_exc()
            return render(request, 'error.html', {
                'error': 'Invalid input',
                'details': str(e)
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return render(request, 'error.html', {
                'error': 'Prediction error',
                'details': str(e)
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def api_predict(request):
    """
    API endpoint for predictions (JSON response) - Updated for V2
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            rank = int(data.get('rank'))
            category = data.get('category', 'OPEN')
            gender = data.get('gender', 'Male')
            home_state = data.get('home_state', None)
            preferences = data.get('preferences', {})
            limit = int(data.get('limit', 50))
            
            if rank <= 0 or rank > 500000:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid rank'
                }, status=400)
            
            if not predictor:
                return JsonResponse({
                    'success': False,
                    'error': 'Model not available'
                }, status=500)
            
            predictions = predictor.predict_colleges(
                rank=rank,
                category_rank=category,
                gender=gender,
                home_state=home_state,
                preferences=preferences
            )
            
            return JsonResponse({
                'success': True,
                'rank': rank,
                'category': category,
                'gender': gender,
                'home_state': home_state,
                'total_results': len(predictions),
                'predictions': predictions[:limit]
            })
            
        except Exception as e:
            print(f"API Error: {e}")
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def get_unique_values(column):
    """
    Helper function to get unique values from institutes data
    """
    if predictor and predictor.institutes_data is not None:
        try:
            values = sorted(predictor.institutes_data[column].unique().tolist())
            return values[:100]  # Limit to 100 for performance
        except Exception as e:
            print(f"Error getting unique values for {column}: {e}")
            return []
    return []


def statistics(request):
    """
    Display statistics about the dataset with V2 data structure
    """
    if not predictor or predictor.institutes_data is None:
        return render(request, 'error.html', {
            'error': 'Data not available'
        })
    
    df = predictor.institutes_data
    
    # Category statistics
    categories_data = []
    if 'seat_type' in df.columns:
        avg_opening = df.groupby('seat_type')['opening_rank'].mean().to_dict()
        avg_closing = df.groupby('seat_type')['closing_rank'].mean().to_dict()
        category_counts = df['seat_type'].value_counts().to_dict()
        
        for category in category_counts.keys():
            opening = avg_opening.get(category, 0)
            closing = avg_closing.get(category, 0)
            
            # Determine competition level
            if opening < 5000:
                competition = 'Very High'
                badge_class = 'high'
                icon = '🥇'
            elif opening < 20000:
                competition = 'High'
                badge_class = 'medium'
                icon = '🥈'
            else:
                competition = 'Moderate'
                badge_class = 'low'
                icon = '🥉'
            
            categories_data.append({
                'name': category,
                'count': category_counts[category],
                'avg_opening': int(opening),
                'avg_closing': int(closing),
                'range': int(closing - opening),
                'competition': competition,
                'badge_class': badge_class,
                'icon': icon
            })
        
        # Sort categories by opening rank
        categories_data.sort(key=lambda x: x['avg_opening'])
    
    # State-wise statistics
    states_data = []
    if 'institute_state' in df.columns:
        state_counts = df['institute_state'].value_counts().to_dict()
        for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            if state != 'Unknown':
                states_data.append({
                    'name': state,
                    'count': count,
                    'percentage': int((count / len(df)) * 100)
                })
    
    # Quota statistics
    quota_stats = []
    if 'quota' in df.columns:
        quota_counts = df['quota'].value_counts().to_dict()
        for quota, count in quota_counts.items():
            quota_stats.append({
                'name': 'All India' if quota == 'AI' else 'Home State',
                'code': quota,
                'count': count,
                'percentage': int((count / len(df)) * 100)
            })
    
    # Gender statistics
    gender_stats = []
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts().to_dict()
        for gender, count in gender_counts.items():
            gender_stats.append({
                'name': gender,
                'count': count,
                'percentage': int((count / len(df)) * 100)
            })
    
    # Top institutes
    institutes_list = []
    max_institute_count = 0
    if 'institute' in df.columns:
        for institute, count in df['institute'].value_counts().head(10).items():
            if max_institute_count == 0:
                max_institute_count = count
            institutes_list.append({
                'name': institute,
                'count': count,
                'percentage': int((count / max_institute_count) * 100)
            })
    
    # Top programs
    programs_list = []
    max_program_count = 0
    if 'program_name' in df.columns:
        for program, count in df['program_name'].value_counts().head(10).items():
            if max_program_count == 0:
                max_program_count = count
            programs_list.append({
                'name': program,
                'count': count,
                'percentage': int((count / max_program_count) * 100)
            })
    
    context = {
        'total_programs': len(df),
        'total_institutes': df['institute'].nunique() if 'institute' in df.columns else 0,
        'total_programs_unique': df['program_name'].nunique() if 'program_name' in df.columns else 0,
        'total_states': df['institute_state'].nunique() if 'institute_state' in df.columns else 0,
        'categories_data': categories_data,
        'states_data': states_data,
        'quota_stats': quota_stats,
        'gender_stats': gender_stats,
        'institutes_list': institutes_list,
        'programs_list': programs_list,
    }
    
    return render(request, 'statistics.html', context)


def test(request):
    """
    Test endpoint to verify setup
    """
    info = {
        'model_loaded': predictor is not None,
        'data_loaded': predictor.institutes_data is not None if predictor else False,
        'model_url': MODEL_URL,
        'version': 'v2'
    }
    
    if predictor and predictor.institutes_data is not None:
        df = predictor.institutes_data
        info['total_records'] = len(df)
        info['institutes'] = df['institute'].nunique() if 'institute' in df.columns else 0
        info['programs'] = df['program_name'].nunique() if 'program_name' in df.columns else 0
        info['states'] = df['institute_state'].nunique() if 'institute_state' in df.columns else 0
        info['quotas'] = df['quota'].unique().tolist() if 'quota' in df.columns else []
        info['seat_types'] = df['seat_type'].unique().tolist() if 'seat_type' in df.columns else []
    
    return JsonResponse(info)


def compare_colleges(request):
    """
    Compare multiple colleges side by side
    """
    if request.method == 'POST':
        try:
            college_ids = request.POST.getlist('college_ids[]')
            rank = int(request.POST.get('rank'))
            category = request.POST.get('category', 'OPEN')
            
            if not predictor or predictor.institutes_data is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Model not loaded'
                }, status=500)
            
            # Get data for selected colleges
            df = predictor.institutes_data
            comparisons = []
            
            for college_id in college_ids[:5]:  # Limit to 5 colleges
                college_data = df[df['institute'] == college_id]
                if len(college_data) > 0:
                    avg_opening = college_data['opening_rank'].mean()
                    avg_closing = college_data['closing_rank'].mean()
                    
                    comparisons.append({
                        'name': college_id,
                        'programs_count': len(college_data),
                        'avg_opening': int(avg_opening),
                        'avg_closing': int(avg_closing),
                        'your_rank': rank,
                        'in_range': avg_opening <= rank <= avg_closing
                    })
            
            return JsonResponse({
                'success': True,
                'comparisons': comparisons
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)