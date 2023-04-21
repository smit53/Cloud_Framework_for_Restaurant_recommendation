from django.shortcuts import render
from django.http import HttpResponse


def recommendation(request):
    return render(request, 'foodapp/recommendation.html')

