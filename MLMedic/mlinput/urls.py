from django.urls import path
from mlinput.views import *

urlpatterns = [
    path('', predict_heartattack, name='base_model'),
    path('answer/', answer_heartattack, name='answer'),
    path('test/', test, name='test')
]
