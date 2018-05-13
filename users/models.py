from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractUser, BaseUserManager


class CustomUserManager(BaseUserManager):
    pass    

class CustomUser(AbstractUser):
    objects = CustomUserManager()