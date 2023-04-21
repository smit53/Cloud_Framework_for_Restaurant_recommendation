from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='food-home'),
    path('location/', views.location, name='location'),
]