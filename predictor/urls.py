from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('history/', views.history_view, name='history'),
    path('api/predict/', views.api_predict, name='api_predict'),
]
