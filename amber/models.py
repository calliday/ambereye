from django.db import models

class TimestampOnUpdateQuerySet(models.QuerySet):
    def update(self, **kwargs):
        # Insert the update timestamp into bulk-updates so we don't have to remember to do this every time ourselves.
        kwargs['updated_at'] = timezone.now()
        super().update(**kwargs)

class TimestampOnUpdateManager(models.Manager):
    def get_queryset(self):
        return TimestampOnUpdateQuerySet(self.model)

class LowercaseEmailField(models.EmailField):
    """
    An email field that ensures its input is lowercase. This is better than overriding `save()`
    because `save()` is not called on bulk updates.
    """

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value:
            value = value.lower()
        return value

class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, editable=False, db_index=True, verbose_name='Created')
    updated_at = models.DateTimeField(auto_now=True, editable=False, db_index=True, verbose_name='Updated')

    objects = TimestampOnUpdateManager()

    revision_message_ignore = ['created_at', 'updated_at']

    @classmethod
    def from_db(cls, db, field_names, values):
        instance = super().from_db(db, field_names, values)
        instance._loaded_values = dict(zip(field_names, values))
        return instance

    # def save(self, *args, **kwargs):
        # if reversion.is_active():
        #     if not reversion.get_comment():
        #         changes = []
        #         if not getattr(self, '_loaded_values', None):
        #             changes.append({'added': {}})
        #         else:
        #             fields = [
        #                 k for k in self._loaded_values
        #                 if k not in self.revision_message_ignore and self._loaded_values[k] != getattr(self, k, None)]
        #             changes.append({'changed': {'fields': fields}})

        #         reversion.set_comment(json.dumps(changes))
        # super().save(*args, **kwargs)

    class Meta:
        abstract = True


class Car(BaseModel):
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
    license_plate = models.CharField(max_length=20)
    style = models.CharField(max_length=10, choices=STYLES)

    def __str__(self):
        return "{} {}".format(self.color, self.style)


class CarPlacement(BaseModel):
    car = models.ForeignKey('Car', on_delete=models.CASCADE)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
