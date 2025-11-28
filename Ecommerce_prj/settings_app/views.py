from rest_framework import viewsets
from rest_framework.permissions import IsAdminUser
from .models import Setting
from .serializers import SettingSerializer

class SettingViewSet(viewsets.ModelViewSet):
    queryset = Setting.objects.all()
    serializer_class = SettingSerializer
    permission_classes = [IsAdminUser]
