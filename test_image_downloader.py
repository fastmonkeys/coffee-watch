import os
import requests
import time

s = requests.Session()
s.auth = (
    os.environ.get('CAMERA_USERNAME', 'admin'),
    os.environ.get('CAMERA_PASSWORD', '')
)
img_url = os.environ.get(
    'CAMERA_IMAGE_URL',
    'http://192.168.179.4/cgi/jpg/image.cgi'
)

response = s.get(img_url)
if not response.ok:
    print response
else:
    filename = time.strftime('sample_images/%d-%m_%H%M%S.jpg')
    with open(filename, 'wb') as f:
        f.write(response.content)
