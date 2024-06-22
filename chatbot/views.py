from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .chatbot_logic import get_response_for_query
# from django.http import HttpResponse
from django.http import JsonResponse
# Create your views here.
@csrf_exempt 
def chat(request):
        if request.method == 'POST':
         message = request.POST.get('message')
         response_data = get_response_for_query(message)
         print(message)
         if isinstance(response_data, str) and response_data.startswith('iVBOR'):  # Check if response is a base64 image
            return JsonResponse({'image': response_data})
         else:
            return JsonResponse({'message': response_data})
        #  return JsonResponse(data,safe=False)
        elif request.method == 'GET':
         return render(request,'home.html')
        else:
         print('Only POST and GET method is allowed')
         return JsonResponse({'message':'error'})
def contact(request):
    return render(request,'contact.html')
