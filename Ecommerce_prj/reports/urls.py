from django.urls import path
from .views import SalesReport

urlpatterns = [
    path('sales/', SalesReport.as_view()),
]
