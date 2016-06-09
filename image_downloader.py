import os
import sys
import cv2
import numpy
import requests
import socket
from time import sleep
from datetime import datetime

from image_processor import process_image

s = requests.Session()
s.auth = (
    os.environ.get('CAMERA_USERNAME', 'admin'),
    os.environ.get('CAMERA_PASSWORD', '')
)
img_url = os.environ.get(
    'CAMERA_IMAGE_URL',
    'http://192.168.179.4/cgi/jpg/image.cgi'
)
from server import db, Measurement

SUCCESS_INTERVAL = int(os.environ.get('SUCCESS_INTERVAL', '30'))
FAILURE_INTERVAL = int(os.environ.get('FAILURE_INTERVAL', '2'))


def send_measurement(value):
    db.session.add(Measurement(
        value=value,
        timestamp=datetime.now()
    ))
    db.session.commit()


if __name__ == "__main__":
    window = '-window' in sys.argv

    while(True):
        try:
            response = s.get(img_url, timeout=5)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            socket.timeout
        ):
            print("Timeout")
            sleep(1)
            continue
        if not response.ok:
            print(response)
            continue
        buf = numpy.frombuffer(response.content, dtype="int8")
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img, value = process_image(img, 'memory/memory.jpg')
        if window:
            cv2.imshow('image', img)
        if value is not None:
            send_measurement(value)
            sleep(SUCCESS_INTERVAL)
        else:
            sleep(FAILURE_INTERVAL)
    if window:
        cv2.destroyAllWindows()
