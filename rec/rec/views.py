from django.shortcuts import render
from . import inference

def home(request):
    return render(request, 'index.html')

def result(request):
    user_id_input = int(request.GET["user_id"])
    item_id_input = int(request.GET["item_id"])
    prediction = inference.predicted_rating(user_id_input, item_id_input)
    return render(request, 'result.html', {'prediction':prediction})
