import cv2
import numpy
import requests
import time

from stuff import process_image

s = requests.Session()
s.auth = ('admin', '')

response = s.get('http://192.168.179.4/cgi/jpg/image.cgi')
if not response.ok:
    print response
else:
    filename = time.strftime('temp3/%d-%m_%H%M%S.jpg')
    with open(filename, 'wb') as f:
        f.write(response.content)
