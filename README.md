# JEE College Predictor API

A Django-based REST API that predicts college admission chances for JEE (Joint Entrance Examination) candidates based on their rank, category, gender, and home state using a trained Machine Learning model.

## Overview

The JEE College Predictor uses a Random Forest classifier trained on historical JEE admission data to predict which colleges and programs a student is likely to get admission into. The system considers multiple factors including:

- **JEE Rank**: The student's category-specific rank
- **Category**: Seat type (OPEN, OBC-NCL, SC, ST, EWS, PwD categories)
- **Gender**: Male or Female (affects Gender-Neutral and Female-only seat eligibility)
- **Home State**: For Home State (HS) quota predictions
- **Preferences**: Optional filters for specific institutes, programs, or states

## Features

- **Intelligent Predictions**: Categorizes admission chances into High, Good, Moderate, Low, and Very Low
- **Quota-Aware**: Handles both All India (AI) and Home State (HS) quotas
- **Gender-Sensitive**: Correctly applies Gender-Neutral and Female-only seat rules
- **Flexible Filtering**: Filter by institute, program, or state preferences
- **Confidence Scoring**: Provides confidence percentages for each prediction
- **Auto-Loading Model**: Downloads and caches the ML model from GitHub releases

## Installation

### Prerequisites

- Python 3.8+
- Django 3.2+
- Required packages: `scikit-learn`, `pandas`, `joblib`, `requests`

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver
```

## API Endpoints

### 1. Health Check

Check if the API and model are loaded correctly.

**Endpoint:** `GET /api/health/`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_available": true
}
```

---

### 2. Get Available Options

Retrieve available categories, genders, and states for the prediction form.

**Endpoint:** `GET /api/options/`

**Response:**
```json
{
  "success": true,
  "categories": [
    "OPEN",
    "OBC-NCL",
    "SC",
    "ST",
    "EWS",
    "OPEN (PwD)",
    "OBC-NCL (PwD)",
    "SC (PwD)",
    "ST (PwD)",
    "EWS (PwD)"
  ],
  "genders": ["Male", "Female"],
  "states": [
    "Andhra Pradesh",
    "Delhi",
    "Karnataka",
    "Maharashtra",
    "Tamil Nadu",
    ...
  ]
}
```

---

### 3. Get Predictions

Predict college admission chances based on user inputs.

**Endpoint:** `POST /api/predict/`

**Request Body:**
```json
{
  "rank": 10000,
  "category": "OPEN",
  "gender": "Male",
  "home_state": "Delhi",
  "preferences": {   # optional
    "institute": "IIT Delhi",
    "program": "Computer Science",
    "state": "Delhi"
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `rank` | integer | Yes | JEE rank (1-500000) |
| `category` | string | Yes | Seat type category |
| `gender` | string | Yes | "Male" or "Female" |
| `home_state` | string | No | Home state for HS quota |
| `preferences` | object | No | Additional filters |
| `preferences.institute` | string | No | Filter by institute name |
| `preferences.program` | string | No | Filter by program name |
| `preferences.state` | string | No | Filter by institute state |

**Response:**
```json
{
  "success": true,
  "rank": 10000,
  "category": "OPEN",
  "gender": "Male",
  "home_state": "Delhi",
  "total_predictions": 150,
  "predictions": {
    "high_chance": [
      {
        "institute": "National Institute of Technology, Delhi",
        "program": "Computer Science and Engineering",
        "quota": "HS",
        "seat_type": "OPEN",
        "gender": "Gender-Neutral",
        "institute_state": "Delhi",
        "opening_rank": 8500,
        "closing_rank": 12000,
        "your_rank": 10000,
        "admission_probability": 75.5,
        "confidence": 75.5,
        "status": "High Chance"
      }
    ],
    "good_chance": [...],
    "moderate_chance": [...],
    "low_chance": [...],
    "all": [...]
  }
}
```

**Prediction Categories:**

- **High Chance** (90-60% confidence): Rank in top 33% of admission range
- **Good Chance** (60-40% confidence): Rank in middle 33% of admission range
- **Moderate Chance** (40-20% confidence): Rank in bottom 33% of admission range
- **Low Chance** (<40% confidence): Rank slightly beyond closing rank
- **Very Low Chance** (<20% confidence): Rank well beyond closing rank

---

### 4. Debug Data

Inspect the loaded dataset (useful for troubleshooting).

**Endpoint:** `GET /api/debug/`

**Response:**
```json
{
  "success": true,
  "total_records": 15000,
  "columns": ["institute", "program_name", "quota", "seat_type", ...],
  "sample_data": [...],
  "unique_values": {
    "seat_type": {
      "count": 10,
      "values": ["OPEN", "OBC-NCL", "SC", "ST", ...]
    }
  },
  "seat_type_distribution": {
    "OPEN": 5000,
    "OBC-NCL": 3000,
    ...
  }
}
```

## Usage Examples

### Example 1: Male Student, OPEN Category

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "rank": 5000,
    "category": "OPEN",
    "gender": "Male",
    "home_state": "Karnataka"
  }'
```

### Example 2: Female Student, OBC-NCL Category with Preferences

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "rank": 15000,
    "category": "OBC-NCL",
    "gender": "Female",
    "home_state": "Maharashtra",
    "preferences": {  # OPTIONAL
      "program": "Computer Science",
      "state": "Maharashtra"
    }
  }'
```

### Example 3: SC Category Student Looking for IITs

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "rank": 8000,
    "category": "SC",
    "gender": "Male",
    "home_state": "Delhi",
    "preferences": {  # OPTIONAL
      "institute": "IIT"
    }
  }'
```

## How It Works

### Data Processing

1. **Normalization**: Standardizes column names and values (e.g., GEN → OPEN)
2. **State Extraction**: Identifies institute states from names using keyword mapping
3. **Encoding**: Converts categorical variables to numerical format for ML model

### Prediction Logic

The system evaluates admission chances based on:

1. **Rank Comparison**: Compares user rank with opening and closing ranks
2. **Quota Filtering**: 
   - All India (AI) quota: Available to all students
   - Home State (HS) quota: Only for students from that state
3. **Gender Rules**:
   - Male: Only Gender-Neutral seats
   - Female: Both Gender-Neutral and Female-only seats
4. **Confidence Calculation**: Position within admission range determines confidence

### Rank Logic (Important!)

In JEE, **lower rank is better**. Admission logic:

- **Rank < Opening Rank**: Excellent chance (98% confidence)
- **Opening ≤ Rank ≤ Closing**: Good to High chance (60-90% confidence)
- **Rank > Closing**: Low to Very Low chance (<40% confidence)

## Model Information

### Training Data

The model is trained on historical JEE admission data containing:
- Institute names and locations
- Program/branch names
- Quota types (AI/HS)
- Seat types (categories)
- Gender requirements
- Opening and closing ranks

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Features**: rank, seat_type, gender, quota, institute, program
- **Training Strategy**: Synthetic samples at multiple rank positions within ranges
- **Accuracy**: Typically ~85-90% on test data

## Error Handling

### Common Errors

**Invalid Rank:**
```json
{
  "success": false,
  "error": "Invalid rank. Please enter a rank between 1 and 500000"
}
```

**Model Not Loaded:**
```json
{
  "success": false,
  "error": "Model not loaded"
}
```

**Invalid JSON:**
```json
{
  "success": false,
  "error": "Invalid JSON format"
}
```

### No Predictions Found

If no matching programs are found, the API returns debug information:

```json
{
  "success": true,
  "total_predictions": 0,
  "predictions": {...},
  "debug_info": {
    "message": "No matching programs found",
    "available_categories": ["OPEN", "SC", ...],
    "suggestion": "Try using OPEN category or check category name"
  }
}
```

## Configuration

### Environment Variables

- `MODEL_URL`: URL to download the ML model (defaults to GitHub release)

### URL Configuration

Add to your Django `urls.py`:

```python
from django.urls import path, include

urlpatterns = [
    path('', include('predictor.urls')),
]
```

## Limitations

- Predictions are based on historical data and may not reflect current year trends
- Rank range is limited to 1-500,000
- Requires stable internet connection for initial model download
- Model accuracy depends on training data quality and completeness

## Contributing

To improve the model:

1. Add more recent admission data to the training dataset
2. Retrain the model using `jee_predictor.py`
3. Upload the new model to GitHub releases
4. Update the `MODEL_URL` environment variable

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
- Check the `/api/debug/` endpoint for data inspection
- Review server logs for detailed error traces
- Ensure the model URL is accessible and the file is valid

---

**Last Updated:** January 2026  
**Model Version:** v2.0  
**API Version:** 1.0
