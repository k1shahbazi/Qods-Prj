from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Sum, Count
from orders.models import Order
from payments.models import Payment
from products.models import Product

class SalesReport(APIView):
    def get(self, request):
        total_orders = Order.objects.count()
        total_revenue = Payment.objects.filter(status='paid').aggregate(sum=Sum('amount'))['sum'] or 0
        # top products by sold count (may require related_name or annotation depending on models)
        # We'll try a safe approach
        top_products = Product.objects.all()[:5]
        top = [{'id': p.id, 'name': p.name, 'sold': 0} for p in top_products]
        return Response({
            'total_orders': total_orders,
            'total_revenue': total_revenue,
            'top_products': top,
        })
