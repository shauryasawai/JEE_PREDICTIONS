# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Health check
    path('api/health/', views.health, name='health'),
    path('api/options/', views.get_options, name='get_options'),    
    path('api/predict/', views.predict, name='predict')
    ]