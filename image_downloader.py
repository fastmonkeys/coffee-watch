import cv2
import numpy
import requests

from stuff import process_image

s = requests.Session()
s.auth = ('admin', '')

while(True):
    response = s.get('http://192.168.179.4/cgi/jpg/image.cgi')
    if not response.ok:
        print response
        continue
    buf = numpy.frombuffer(response.content, dtype="int8")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = process_image(img, 'memory/memory.jpg')
    cv2.imshow('image', img)
    if cv2.waitKey(1) != -1:
        break
cv2.destroyAllWindows()
