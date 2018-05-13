from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('dash/', views.dash.as_view(), name='dash'),
    path('portfolio/', views.portfolio.as_view(), name='portfolio'),
    path('tesla/', views.tesla, name='tesla'),
    path('passchange/', views.passchange.as_view(), name='passchange'),
    path('passchangedone/', views.passchangedone.as_view(), name='passchangedone'),
    path('airtel/', views.airtel, name='airtel'),
    path('apple/', views.apple, name='apple'),
    path('amazon/', views.amazon, name='amazon'),
    path('google/', views.google, name='google'),
    path('microsoft/', views.microsoft, name='microsoft'),
    path('sensex/', views.sensex, name='sensex'),
    
]