from django.core.management.base import BaseCommand

# import the necessary packages
from amber.models import Car, CarPlacement
from amber.cleaned_yolo import run


class Command(BaseCommand):
    help = 'Start capturing images'
    
    def handle(self, **options):
        run()