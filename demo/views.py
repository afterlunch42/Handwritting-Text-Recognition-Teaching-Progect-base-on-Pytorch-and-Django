from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from predict_demo import recognize_input_image


def index(request):
    print(1)
    return render(request, 'home.html')


@csrf_exempt
def process(request):
    print(2)
    if (request.method == "POST") and (request.POST.get('id') == "1"):
        imgStr = request.POST.get('txt')
        print(imgStr)
        # imgStr.replace(" ", "+")
        imgStr = base64.b64decode(imgStr)
    # 识别结果
        prob, pred = recognize_input_image(imgStr)  # TODO:注意修改此处的传入参数(原方法传入图片路径)
        return HttpResponse(json.dumps({"status": 1, "prob": prob, "pred": pred}))
    else:
        return HttpResponse(json.dumps({"status": -1, "prob":"", "pred": ""}))