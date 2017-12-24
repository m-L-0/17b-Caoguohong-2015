from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from django.shortcuts import render
import subprocess
import requests


def yan(request):
    return render(request, 'index.html')


def upload(request):
    handle_uploaded_file(str(request.FILES['file-zh-TW[]']), request.FILES['file-zh-TW[]'])
    file = './static/upload/' + str(request.FILES['file-zh-TW[]'])
    a = subprocess.getoutput("./tools/predict "+file)
    result = str(a).split('\n')[-1]
    return HttpResponse("{\"extra\": \"" + result + "\"}")


def handle_uploaded_file(filename, f):
    with open('./static/upload/' + filename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)



def api(request):
    imgurl=request.GET['url']
    imgdata = requests.get(imgurl).content
    with open('./static/upload/' + 'temp.jpg', 'wb') as f:
        f.write(imgdata)
    file='./static/upload/temp.jpg'
    a = subprocess.getoutput("./tools/predict " + file)
    result = str(a).split('\n')[-1]
    return HttpResponse('識別結果為'+result)