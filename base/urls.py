# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/health/', views.health, name='health'),
    path('api/debug/', views.debug_data, name='debug_data'),
    path('api/options/', views.get_options, name='get_options'),
    path('api/predict/', views.predict, name='predict'),
]