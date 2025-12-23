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
from .jee_predictor import JEEPredictor

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Initialize predictor globally
predictor = None

# Model URL from GitHub Release
MODEL_URL = os.environ.get(
    'MODEL_URL',
    'https://github.com/shauryasawai/JEE_PREDICTIONS/releases/download/v1.0.0-JEE_prediction/jee_predictor_model.pkl'
)

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
            print("Initializing predictor...")
            predictor = JEEPredictor()
            
            # Download and load model from GitHub
            model_data = download_and_load_model()
            
            # Load the model data into predictor
            predictor.model = model_data.get('model')
            predictor.institutes_data = model_data.get('institutes_data')
            predictor.scaler = model_data.get('scaler')
            
            if predictor.institutes_data is None:
                raise Exception("Model data is corrupted or incomplete")
            
            print(f"Predictor loaded with {len(predictor.institutes_data)} records")
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
    
    context = {
        'categories': ['GEN', 'OBC-NCL', 'SC', 'ST', 'GEN-PWD', 'OBC-NCL-PWD', 'SC-PWD', 'ST-PWD'],
        'institutes': get_unique_values('institute_short'),
        'programs': get_unique_values('program_name'),
        'degrees': get_unique_values('degree_short'),
    }
    return render(request, 'index.html', context)


def predict(request):
    """
    Handle prediction requests - OPTIMIZED
    """
    if request.method == 'POST':
        try:
            # Debug: Print received data
            print("\n" + "="*80)
            print("PREDICTION REQUEST RECEIVED")
            print("="*80)
            
            # Get form data
            rank = int(request.POST.get('rank'))
            category = request.POST.get('category')
            
            print(f"Rank: {rank}")
            print(f"Category: {category}")
            
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
            degree_filter = request.POST.get('degree_filter')
            
            if institute_filter and institute_filter != 'all':
                preferences['institute_short'] = institute_filter
            if program_filter and program_filter != 'all':
                preferences['program_name'] = program_filter
            if degree_filter and degree_filter != 'all':
                preferences['degree_short'] = degree_filter
            
            print(f"Preferences: {preferences}")
            print("Starting prediction...")
            
            # Get predictions
            predictions = predictor.predict_colleges(rank, category, preferences)
            
            print(f"Total predictions: {len(predictions)}")
            
            # Categorize predictions
            high_chance = [p for p in predictions if p['status'] == 'High Chance']
            good_chance = [p for p in predictions if p['status'] == 'Good Chance']
            moderate_chance = [p for p in predictions if p['status'] == 'Moderate Chance']
            low_chance = [p for p in predictions if p['status'] in ['Low Chance', 'Very Low Chance']]
            
            print(f"High: {len(high_chance)}, Good: {len(good_chance)}, Moderate: {len(moderate_chance)}, Low: {len(low_chance)}")
            print("="*80 + "\n")
            
            context = {
                'rank': rank,
                'category': category,
                'total_predictions': len(predictions),
                'high_chance': high_chance[:20],
                'good_chance': good_chance[:20],
                'moderate_chance': moderate_chance[:20],
                'low_chance': low_chance[:10],
                'all_predictions': predictions[:50],
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
    API endpoint for predictions (JSON response)
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            rank = int(data.get('rank'))
            category = data.get('category')
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
            
            predictions = predictor.predict_colleges(rank, category, preferences)
            
            return JsonResponse({
                'success': True,
                'rank': rank,
                'category': category,
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
    Display statistics about the dataset
    """
    if not predictor or predictor.institutes_data is None:
        return render(request, 'predictor/error.html', {
            'error': 'Data not available'
        })
    
    df = predictor.institutes_data
    
    # Pre-process data for easier template rendering
    categories_data = []
    avg_opening = df.groupby('category')['opening_rank'].mean().to_dict()
    avg_closing = df.groupby('category')['closing_rank'].mean().to_dict()
    category_counts = df['category'].value_counts().to_dict()
    
    for category in category_counts.keys():
        opening = avg_opening.get(category, 0)
        closing = avg_closing.get(category, 0)
        
        # Determine competition level
        if opening < 5000:
            competition = 'Very High'
            badge_class = 'high'
        elif opening < 20000:
            competition = 'High'
            badge_class = 'medium'
        else:
            competition = 'Moderate'
            badge_class = 'low'
        
        categories_data.append({
            'name': category,
            'count': category_counts[category],
            'avg_opening': int(opening),
            'avg_closing': int(closing),
            'range': int(closing - opening),
            'competition': competition,
            'badge_class': badge_class,
            'icon': '🥇' if opening < 5000 else '🥈' if opening < 20000 else '🥉'
        })
    
    # Sort categories by opening rank
    categories_data.sort(key=lambda x: x['avg_opening'])
    
    # Get top institutes and programs
    institutes_list = []
    max_institute_count = 0
    for institute, count in df['institute_short'].value_counts().head(10).items():
        if max_institute_count == 0:
            max_institute_count = count
        institutes_list.append({
            'name': institute,
            'count': count,
            'percentage': int((count / max_institute_count) * 100)
        })
    
    programs_list = []
    max_program_count = 0
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
        'total_institutes': df['institute_short'].nunique(),
        'total_programs_unique': df['program_name'].nunique(),
        'categories_data': categories_data,
        'institutes_list': institutes_list,
        'programs_list': programs_list,
    }
    
    return render(request, 'statistics.html', context)


# Test view to check if everything is working
def test(request):
    """
    Test endpoint to verify setup
    """
    info = {
        'model_loaded': predictor is not None,
        'data_loaded': predictor.institutes_data is not None if predictor else False,
        'model_url': MODEL_URL,
    }
    
    if predictor and predictor.institutes_data is not None:
        info['total_records'] = len(predictor.institutes_data)
        info['institutes'] = predictor.institutes_data['institute_short'].nunique()
        info['programs'] = predictor.institutes_data['program_name'].nunique()
    
    return JsonResponse(info)