from django.db import models

class Car(models.Model):
	STYLES = (
		('suv', 'SUV'),
		('sed', 'Sedan'),
		('coup', 'Coupe'),
		('semi', 'Semi'),
		('pick', 'Pickup'),
		('moto', 'Motorcycle'),
		('bus', 'Bus'),
	)
	color = models.CharField(max_length=20)
	lic = models.CharField(max_length=20)
	style = models.CharField(max_length=10, choices=STYLES)
