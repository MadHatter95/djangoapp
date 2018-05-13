from django.shortcuts import render
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.dates import DateFormatter 
# Create your views here.
from django.urls import reverse_lazy
from django.views import generic
from .nnprediction import prediction,predictapple,predictairtel,predictamazon,predictgoogle,predictmicrosoft,predictsensex
from .forms import CustomUserCreationForm
from random import randint
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
#import matplotlib matplotlib.use('Agg')
from matplotlib.figure import Figure 
from chartjs.views.lines import BaseLineChartView
from django.views.generic import TemplateView
import plotly.offline as opy
import plotly.graph_objs as go





class SignUp(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'


class dash(generic.TemplateView):
    template_name = 'dash.html' 

class portfolio(generic.TemplateView):
    template_name = 'portfolio.html'   


class passchange(generic.TemplateView):
    template_name = 'registration/passchange.html'

class passchangedone(generic.TemplateView):
    template_name = 'registration/password_change_done.html'        


    
def tesla(request):
    p = prediction()
    x,y=p.fun()
    context={'x':x, 'y':y}
    
    return render(request,'tesla.html',context)


def airtel(request):
    artl = predictairtel()
    x2,y2=artl.funairtel()
    artcont={'x2':x2,'y2':y2}
    return render(request,'airtel.html',artcont)

def amazon(request):
    amzn = predictamazon()
    x3,y3=amzn.funamzn()
    amzcont={'x3':x3,'y3':y3}
    return render(request,'amazon.html',amzcont)


def apple(request):
    
    ap = predictapple()
    x1,y1 = ap.funapple()
    apcont={'x1':x1,'y1':y1}
    return render(request,'apple.html',apcont)

def google(request):
    go = predictgoogle()
    x4,y4 = go.fungoogle()
    gocont = {'x4':x4,'y4':y4}
    return render(request,'google.html',gocont)

def microsoft(request):
    mic=predictmicrosoft()
    x5,y5 = mic.funmsft()
    miccont={'x5':x5,'y5':y5}
    return render(request,'microsoft.html',miccont)
    
def sensex(request):
    sen=predictsensex()
    x6,y6 = sen.funsensex()
    sencont={'x6':x6,'y6':y6}
    return render(request,'sensex.html',sencont)

