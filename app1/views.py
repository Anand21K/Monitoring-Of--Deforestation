from django.shortcuts import render
# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from fastai.vision.all import *

import json
import base64
import os


def home(request):
    return HttpResponse("Hello, this is your Django app!")

# Replace with the appropriate HTTP methods
@csrf_exempt
def index(request):
    if request.method == 'POST':
        # print(request.body)
        data = request.body
        json_data = json.loads(data)
        image = json_data.get('imageBuffer').replace("data:image/jpeg;base64,","" )
        print(os.getcwd())
        flag = save_base64_image(image, './app1/images/photo_1')
        # print("imagefile", image)
        if flag:
            return predict()
        else:
            return JsonResponse({"message": "Failed"})


def save_base64_image(base64_string, file_path):
    """
    Save a base64 encoded image to a file.

    Args:
        base64_string (str): The base64 encoded image data.
        file_path (str): The path where the image file will be saved.

    Returns:
        bool: True if the image is successfully saved, False otherwise.
    """
    try:
        # Remove the data:image/{image_extension};base64, part from the base64_string
        image_data = base64_string.split(',')[0]
        # print(image_data)
        # Decode the base64 string to binary data
        binary_data = base64.b64decode(image_data)
        print("hello", os.getcwd())
        # Determine the file extension based on the base64 image header
        file_extension = 'jpeg'
        if not file_extension:
            raise ValueError("Invalid base64 image data")

        # Write the binary data to the image file with the appropriate extension
        with open(f"{file_path}.{file_extension}", 'wb') as image_file:
            image_file.write(binary_data)

        
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False




def predict():
    print(os.getcwd())
    learn = load_learner('../../Downloads/export.pkl')
    learn.eval()
    learn.save('xresnet50_save', with_opt=False)
    # json_response = json.dumps({"prediction ": learn.predict('./app1/images/photo_1.jpeg')[0]})
    response = str(learn.predict('./app1/images/photo_1.jpeg')[0])
    print(response)
    return JsonResponse({"prediction": response})
