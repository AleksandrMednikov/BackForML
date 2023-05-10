#Django
from django.shortcuts import render
from  django.http import HttpResponse
#ML
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np


def predict_heartattack(request):
    params = {
        'model_name': 'Инфаркт',
        'params': [
            {'value': 'age', 'desc': 'Укажите ваш возраст',
             'bool': 0, 'desc_bool_0': None, 'desc_bool_1': None},
            {'value': 'gender', 'desc': 'Укажите ваш пол',
             'bool': 1, 'desc_bool_0': 'Женщина', 'desc_bool_1': 'Мужчина'},
            {'value': 'smoking_alcoholism', 'desc': 'Вы курите или пьете?',
             'bool': 1, 'desc_bool_0': 'Нет', 'desc_bool_1': 'Да'},
            {'value': 'fatness', 'desc': 'У вас есть ожирение?',
             'bool': 1, 'desc_bool_0': 'Нет', 'desc_bool_1': 'Да'},
            {'value': 'diabetes', 'desc': 'Вы страдаете от диабета?',
             'bool': 1, 'desc_bool_0': 'Нет', 'desc_bool_1': 'Да'},
            {'value': 'hyperlipidemia', 'desc': 'У вас есть гиперлипидемия?',
             'bool': 1, 'desc_bool_0': 'Нет', 'desc_bool_1': 'Да'},
            {'value': 'physically_active', 'desc': 'Вы ведёте физически активный образ жизни?',
             'bool': 1, 'desc_bool_0': 'Нет', 'desc_bool_1': 'Да'},
            {'value': 'arterial_hypertension', 'desc': 'У вас есть артериальная гипертензия?',
             'bool': 1, 'desc_bool_0': 'Нет', 'desc_bool_1': 'Да'},
        ]
    }
    if request.GET:
        pass
    else:
        return render(request, 'model_views.html', params)

def answer_heartattack(request):
    model_name = 'Инфаркт'
    threshold = 0.233245
    r_get = request.GET
    x_quest = []

    for i in r_get:
        if i == 'age':
            x_quest.append(int(r_get[i])/110)
        elif i == 'gender':
            x_quest.append(int(r_get[i]))
            if int(r_get[i]) == 1:
                x_quest.append(0)
            elif int(r_get[i]) == 0:
                x_quest.append(1)
        else:
            x_quest.append(int(r_get[i]))

    with open("mlinput/ml_models/knn_model.sav", "rb") as f:
        knn = pickle.load(f)

    with open("mlinput/ml_models/catboost_model.sav", "rb") as f:
        catb = pickle.load(f)

    print([x_quest])
    pred_catb_proba = catb.predict_proba([x_quest])
    pred_catb = catb.predict([x_quest])
    pred_knn = knn.predict([x_quest])

    if pred_catb == 1:
        return render(request, 'model_answer.html',
                      {'model_name': model_name,
                       'desc': 'Вероятность получить инфаркт 100% срочно обратитесь к врачу!',
                       'result': 'Критичная вероятность',
                       'class': 'max'})
    elif pred_catb_proba[:, 1] > threshold:
        return render(request, 'model_answer.html',
                      {'model_name': model_name,
                       'desc': 'Вы близки к инфаркту рекомендуем обратиться к врачу.',
                       'result': 'Высокая вероятность',
                       'class': 'high'})
    elif pred_knn == 1:
        return render(request, 'model_answer.html',
                      {'model_name': model_name,
                       'desc': 'Вы находитесь в зоне риска и с вероятностью 50% можете получить инфаркт.',
                       'result': 'Средняя вероятность',
                       'class': 'average'})
    else:
        return render(request, 'model_answer.html',
                      {'model_name': model_name,
                       'desc': 'Поздравляем вы здоровы!',
                       'result': 'Нет вероятности',
                       'class': 'low'})

def test(request):
    return render(request, 'test.html', {'title': 'my'})
