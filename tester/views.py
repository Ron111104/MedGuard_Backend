# tester/views.py

from django.http import JsonResponse
def hello(request):
    return JsonResponse({"message": "hello"})
