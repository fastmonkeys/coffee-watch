Coffee-watch
============

Coffee-watch is a tool that downloads images of coffee maker from a webcam,
processes the images and lists the current coffee level to a webserver. Image
processing is done using OpenCV and the project only aims to support the coffee
 maker at our office.

Instructions:
-------------

 - Step 0: Install OpenCV

 - Step 1: Setup + start the server:

```
pip install -r requirements.txt
createdb coffee_watch
export DATABASE_URI=postgresql://localhost/coffee_watch
python server.py -create
python server.py
```

Step 2: Start image_downloader

```
export DATABASE_URI=...
export CAMERA_USERNAME=...
export CAMERA_PASSWORD=...
export CAMERA_IMAGE_URL=...
python image_downloader.py
```

Debug images can be found in debug_output folder.
