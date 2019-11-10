from django.contrib import admin
from amber.models import Car, CarPlacement

# Register your models here.
# admin.site.register(Car)
# admin.site.register(CarPlacement)

class BaseAdmin(admin.ModelAdmin):
    # readonly_fields = ('created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at',)

@admin.register(Car)
class CarAdmin(BaseAdmin):
    list_display = (
        'license_plate',
        'color',
        'style',
        'updated_at',
        'created_at',
    )
    search_fields = ['license_plate',]

@admin.register(CarPlacement)
class CarPlacementAdmin(BaseAdmin):
    list_display = (
        'car',
        'latitude',
        'longitude',
        'created_at',
    )
    search_fields = ['license_plate',]
