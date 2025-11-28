from django.db import models

class Setting(models.Model):
    key = models.CharField(max_length=200, unique=True)
    value = models.CharField(max_length=1000)

    def __str__(self):
        return f"{self.key} = {self.value}"
