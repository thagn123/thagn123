from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def home(request):
    return HttpResponse("<h1>Chào mừng đến với hệ thống tư vấn pháp luật!</h1>")
