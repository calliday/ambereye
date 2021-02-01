# Welcome to Project AMBEReye!

This application was developed by Ben Fisher and Caleb Hensley and qualified as their senior project for Brigham Young University - Idaho.

The real-world application of this software is its goal to assist the AMBER alert program in the recovery of children by helping law enforcement in the search for suspected kidnappers. Its purpose is to make a log of car color, type, and location at the point in time that a device 'sees' a car on the road. Human trafficking is a very real and current issue, and this software is a step toward fighting the global battle that it is.

This software is a prototype and is meant to run on a raspberry pi inside a car with an external camera, Intel Neural Compute Stick 2, and GPS receiver.


## Spring 2020 Update
Support was added for an Intel Neural Compute Stick 2. This device cuts our object detection time down by more than 50%.


## Running commands
You must be in `ambereye/` directory to run `python manage.py` commands


## Dev Environment
`python manage.py runserver` to start the server
Go to `localhost:8000/admin` in the browser and login

`python manage.py makemigrations` to set changes to the models
`python manage.py migrate` to migrate model changes to the database


### Initial setup
`python manage.py createsuperuser` to create a user for the backend.
You don't need to add an email.

To run any YOLO detection, you will need to download the weights file from this Google Drive link:
https://drive.google.com/open?id=1Y5nL3_2lZfiirzDjDK3wvLKwAlsIpD0g
