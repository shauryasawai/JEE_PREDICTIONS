# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('statistics/', views.statistics, name='statistics'),
]