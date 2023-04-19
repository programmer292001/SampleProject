import os
import io
import json
import sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
from PIL import Image, ImageDraw, ImageFont
"""
Example 1. Detect faces from an image (from the web)
"""

# credential = json.load(open('AzureCloudKeys.json'))
API_KEY = '966a5ceea7454b81b2bc6566d2bb0d3e'
ENDPOINT = 'https://face-detecter.cognitiveservices.azure.com/'

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
# image_url = 'https://earthyphotography.co.uk/wp-content/uploads/2020/04/composite-corporate-group-photos-02.jpg'
# image_url = '.Training_images\Train.jpg'
# image_url = 'https://media.istockphoto.com/photos/large-group-of-students-studying-on-a-class-in-the-classroom-picture-id1163483094'
# image_url = 'https://p1.piqsels.com/preview/843/292/867/children-students-school-classroom-india-abhaneri.jpg'
image_url='https://lh3.googleusercontent.com/uZT7xUviM3u4xUHayA3AUtg9zOz33K34R0_gqMLGRdjRr9ebvJRT5jyCqqMtKsoyPnWll4Bb7jE_K3TTO_h7u7JtWb7VpWk_mgsqy2itkqJjHYklOWeANMuH0FjMIZYcdK8xq0Xa-41y9MzIV8gMJWUC2ltWyxu1hz6Y-xnYZnrGtZOa1hJPejNHn73kYv1q2rmRknv6CC42ALDV_hqi0tsOqw5XCIdNGS0aT0aR1FnCiTJr17n8nOqGfF5H1tGlOns8nL5KJRGaB039WThTr_eax-qXO9f_dFfGa58U1dUg6wbVlbIJI9t7e-wpdOTBBKjN8bTYSvPA5L5I6XCbAsyHxKK4kqzBe2En7-Y0uH_R-jHwbPlEBib4Bpus3u4UFjRX4WmidD4VCakEAs66seuEbXQjAg8cw2eTlOSErcoh5Q1r2PGNAujwxcJqzLODYyc0pZDqf6JNbTG1E-nfXQFhkoX4Cw9hYgOIlzrRCxJ6ZLjYPUX-yHrL_b3oTxy-OpfyAoEcudOLDesMIPhCsYCM-M9g9XeGjlEawBuwl5PSNBiQpWkKTzHkVoQ1wqiWPFNUUB6wZ7yt7F4og4dT5gyJzYSmnEBEEUoZ7J5ZYtflWeawW751oUQ0qbjO_FNtsJ6rJW4CgV9DU6bhFiDDM8BfgUyLtEDJSo5AdfmNYbL0jOWMQ6HtKdzwgTBoiPuS8rh9tAWTrBhlpoT1rLJXZCjo=w1208-h903-no?authuser=0'
image_name = os.path.basename(image_url)

response_detected_faces = face_client.face.detect_with_url(
    image_url,
    detection_model='detection_03',
    recognition_model='recognition_04'

)
print(response_detected_faces)

if not response_detected_faces:
    raise Exception('No face detected')

print('Number of people detected: {0}'.format(len(response_detected_faces)))

response_image = requests.get(image_url)
img = Image.open(io.BytesIO(response_image.content))
draw = ImageDraw.Draw(img)

for face in response_detected_faces:
    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    draw.rectangle(((left, top), (right, bottom)), outline='green', width=5)
img.show()
img.save('test.jpg')