
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from funcs.ml_funcs import oleg_func_linear
from fastapi.templating import Jinja2Templates
from datetime import datetime
from fastapi.staticfiles import StaticFiles
import os
import base64

app = FastAPI()

@app.get("/model", response_class=HTMLResponse)
async def get_model(request: Request):

    params = [
        {'label': 'Promo 1', 'name': 'promo1'},
        {'label': 'Out_of_stock 1', 'name': 'out_of_stock1'},
        {'label': 'Epidemic 1', 'name': 'epidemic1'},
        {'label': 'Promo 2', 'name': 'promo2'},
        {'label': 'Out_of_stock 2', 'name': 'out_of_stock2'},
        {'label': 'Epidemic 2', 'name': 'epidemic2'},
        {'label': 'Promo 3', 'name': 'promo3'},
        {'label': 'Out_of_stock 3', 'name': 'out_of_stock3'},
        {'label': 'Epidemic 3', 'name': 'epidemic3'},
        {'label': 'Promo 4', 'name': 'promo4'},
        {'label': 'Out_of_stock 4', 'name': 'out_of_stock4'},
        {'label': 'Epidemic 4', 'name': 'epidemic4'},
    ]
    page_answer = "/answer"

    # фаилs по умолчанию
    path_file = (os.listdir(os.getcwd()+"\\read_file")[0]).replace(" ","_").split(".")
    file_type =  path_file[1]
    path_file = path_file[0]

    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("oleg_task.html",
                                      {"request": request,
                                       "params_of_func": params,
                                       "path_file": path_file,
                                       "page_answer": page_answer
                                       })

@app.get("/answer", response_class=HTMLResponse)
async def get_model(request: Request,
                 promo1:int, promo2:int, promo3:int, promo4:int,
                 out_of_stock1:int, out_of_stock2:int, out_of_stock3:int, out_of_stock4:int,
                 epidemic1:int, epidemic2:int, epidemic3:int, epidemic4:int,
                 forecast_periods:int, file_name:str, file_type:str):

    file = "read_file/"+file_name+"."+file_type
    key_now = str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")

    results = oleg_func_linear(file, user_key=key_now,
                 promo1=promo1, promo2=promo2, promo3=promo3, promo4=promo4,
                 out_of_stock1=out_of_stock1, out_of_stock2=out_of_stock2, out_of_stock3=out_of_stock3, out_of_stock4=out_of_stock4,
                 epidemic1=epidemic1, epidemic2=epidemic2, epidemic3=epidemic3, epidemic4=epidemic4,
                 forecast_periods=forecast_periods, file_type=file_type)



    with open(f"results/{results[2]}", "rb") as img_file:
        img_left = base64.b64encode(img_file.read()).decode("utf-8")

    with open(f"results/{results[3]}", "rb") as img_file:
        img_right = base64.b64encode(img_file.read()).decode("utf-8")


    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("answer_for_oleg.html",
                                      {"request": request,
                                       "results0": results[0],
                                       "results1": results[1],
                                       "img_left": img_left,
                                       "img_right": img_right})

    #return "Все успешно, посмотрите результаты в фаиле results"