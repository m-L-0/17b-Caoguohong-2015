from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from django.shortcuts import render


def yan(request):
    return render(request, 'index.html')


def upload(request):
    handle_uploaded_file(str(request.FILES['file-zh-TW[]']), request.FILES['file-zh-TW[]'])
    file = './static/upload/' + str(request.FILES['file-zh-TW[]'])
    print(file)
    result = predict(file)
    return HttpResponse("{\"extra\": \"" + result + "\"}")


def handle_uploaded_file(filename, f):
    with open('./static/upload/' + filename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def predict(filename):
    import cv2
    import numpy
    from keras.models import load_model
    model = load_model('./tools/test1.h5')
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (50, 40))
    img = numpy.resize(img, (1, 1, 40, 50))
    temp = model.predict([img])
    arr = []
    result = ''
    for i in temp:
        arr.append(numpy.argmax(i))
    for i in arr:
        i = str(i)
        if i != '10':
            result += i
    return result
