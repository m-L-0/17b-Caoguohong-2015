from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse


def yan(request):
    return HttpResponse("hello")