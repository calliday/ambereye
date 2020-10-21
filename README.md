# Welcome to Project AMBEReye!

This application was developed by Ben Fisher and Caleb Hensley and qualified as their senior project for Brigham Young University - Idaho.

The real-world application of this software is its goal to assist the AMBER alert program in the recovery of children by helping law enforcement in the search for suspected kidnappers. Human trafficking is a very real and current issue, and this software is a step toward fighting the global battle that it is.

This software is meant to run on a prototype raspberry pi inside a car with an attached camera and GPS receiver.


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
