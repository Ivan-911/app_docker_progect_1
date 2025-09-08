import os
import gc
import random
from datetime import datetime, timedelta

import joblib
import re
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from nltk.corpus import stopwords
import nltk


import requests
import logging
import torch
from sqlalchemy.orm import Session
from typing import Union, Optional
from fastapi import FastAPI, Request, Depends, Body
from fastapi.responses import RedirectResponse
from transformers import BertTokenizer, BertForSequenceClassification
from contextlib import asynccontextmanager

from database import SessionLocal, MyBot, UserRequests, MLPredict, Users, AuthTokens, init_db
from preprocessing import lemmatize_natasha_batch, preprocess_text_batch 

import base64

nltk.download('stopwords')



# Загружаем пайплайн
pipeline = joblib.load('greeting_classifier4.joblib')

HELLO_MESSAGES = [' 🤖 Это автоматическое сообщение от бота. \n Пожалуйста, кратко опишите суть вашего запроса, и мы передадим его компетентному специалисту.',
                  ' 🤖 Вы пишете боту поддержки. \n Уточните, пожалуйста, в чём состоит ваш вопрос, чтобы мы могли оперативно направить его нужному сотруднику.',
                  '🤖 Вы общаетесь с ботом. \n Просим кратко изложить ваш запрос, и мы оперативно переадресуем его специалисту.',
                  ' 🤖 Сообщение сформировано ботом. \n Опишите, пожалуйста, лаконично вашу проблему или запрос для передачи профильному эксперту.',]

def check_message(message: str) -> bool:
    """
    Проверяет, является ли сообщение приветствием.
    
    Args:
        message (str): Сообщение для проверки.
        
    Returns:
        bool: True, если сообщение является приветствием, иначе False.
    """


    prediction = pipeline.predict([message])
    logger.warning(f"Message: {message}, Prediction: {prediction}")
    return prediction[0] == 0

# Путь к вашему файлу
file_path = os.getcwd() + "/Futuristic Logo Design Featuring Geometric Robot Head.png"

with open(file_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')


def predict_label(text, model, tokenizer, categories, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    return categories[predicted_label]


def predict(request_id: int, db: Session):
    """ Predicts the category of a user request using a pre-trained BERT model.

    Args:
        request_id (int): The ID of the user request to predict.
        db (Session): The database session.

    Returns:
        int: The ID of the prediction result.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_dict = os.getcwd() + '/bert_model_final_v2.pth'
    model_name = 'DeepPavlov/rubert-base-cased'
    model_loaded = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model_loaded.load_state_dict(torch.load(path_dict, map_location=device))
    model_loaded.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    categories = ['Консультация на покупку', 'Оплата', 'Техническая поддержка']
    user_request = db.query(UserRequests).filter(UserRequests.request_id == request_id).first()
    predict_category = predict_label(
        user_request.raw_message,
        model=model_loaded,
        tokenizer=tokenizer,
        categories=categories,
        device=device
    )
    ml_predict = MLPredict(
        request_id=user_request.request_id,
        predict_label=predict_category
    )
    db.add(ml_predict)
    db.commit()
    db.refresh(ml_predict)
    # Явно удаляем объекты и чистим память
    del model_loaded
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ml_predict.predict_id

def start_to_process(predict_id: int, db: Session, chat_id: Optional[int] = None):
    ml_predict = db.query(MLPredict).filter(MLPredict.predict_id == predict_id).first()

    if not ml_predict:
        return {"error": "Prediction not found"}
    # Здесь можно добавить логику для обработки предсказания
    user_id = db.query(UserRequests).filter(UserRequests.request_id == ml_predict.request_id).first().user_id
    token = db.query(AuthTokens).filter(AuthTokens.user_id == user_id).order_by(AuthTokens.token_id.desc()).first()

    if ml_predict.predict_label == 'Консультация на покупку':
        oueue_id = 54
    elif ml_predict.predict_label == 'Оплата':
        oueue_id = 56
    elif ml_predict.predict_label == 'Техническая поддержка':
        oueue_id = 52
    else:
        return {"error": "Unknown category"}
        
    if datetime.now() < token.expires_at:
        data = {'QUEUE_ID': oueue_id,
                'CHAT_ID': chat_id,          
                'LEAVE': 'Y',
                'auth': token.access_token}
        response = requests.post("https://sip.bitrix24.kz/rest/imopenlines.bot.session.transfer", json=data)
        logger.warning(f"chat_id: {chat_id}")
        logger.warning(f"Response from imopenlines.bot.session.transfer: {response.text}")
        if response.status_code == 200:
            data = {'CHAT_ID': chat_id,
                    'BOT_ID': 460,
                    'auth': token.access_token,
            }
            response = requests.post("https://sip.bitrix24.kz/rest/imbot.chat.leave", json=data)
            logger.warning(f"Response from imbot.chat.leave: {response.text}")
            logger.warning(f"Workflow started successfully for predict_id {predict_id}")
        else:
            logger.error(f"Failed to start workflow for predict_id {predict_id}: {response.text}")
            return {"error": "Failed to start workflow"}
    else:
        token_url = "https://oauth.bitrix.info/oauth/token/"
        data = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": token.refresh_token,
        }
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            token_data = response.json()
            token.access_token = token_data.get('access_token', '')
            token.refresh_token = token_data.get('refresh_token', '')
            token.expires_at = datetime.now() + timedelta(seconds=int(token_data.get('expires_in', 3600)))
            db.commit()
            data = {'QUEUE_ID': oueue_id,
                'CHAT_ID': chat_id,          
                'LEAVE': 'Y',
                'auth': token.access_token}
            response = requests.post("https://sip.bitrix24.kz/rest/imopenlines.bot.session.transfer", json=data)
            logger.warning(f"chat_id: {chat_id}")
            logger.warning(f"Response from imopenlines.bot.session.transfer: {response.text}")
            if response.status_code == 200:
                data = {'CHAT_ID': chat_id,
                    'BOT_ID': 460,
                    'auth': token.access_token,
                    }
                response = requests.post("https://sip.bitrix24.kz/rest/imbot.chat.leave", json=data)
                logger.warning(f"Response from imbot.chat.leave: {response.text}")
                response = requests.post("https://sip.bitrix24.kz/rest/imbot.chat.leave", json=data)
                logger.warning(f"Response from imbot.chat.leave: {response.text}")
            if response.status_code == 200:
                logger.warning(f"line workflow started successfully for predict_id {predict_id} after token refresh")
            else:
                logger.error(f"Failed to start workflow after token refresh for predict_id {predict_id}: {response.text}")
                return {"error": "Failed to start workflow after token refresh"}    
    return {"message": "Processing started", "predict_id": predict_id}


# --- событие при старте приложения ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    pass

app = FastAPI(lifespan=lifespan)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

path_to_file = os.path.dirname(os.path.abspath(__file__)) + '/loger/fastapi.log'


logging.basicConfig(
    filename=path_to_file,
    filemode='a+',
    #encoding='utf-8',
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - Systems info - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


CLIENT_ID = os.getenv("CLIENT_ID", "local.684ef2bc59c7f8.87517346")
#CLIENT_SECRET = os.getenv("CLIENT_SECRET", "9CWQgq7Da4i46gI5qZkntOKfGX8gT2WwgJFQgF770NX22O4qno")

def red_secret(secret_name):
    try:
        with open(secret_name, 'r') as file:
            secret = file.read().strip()
            return secret
    except FileNotFoundError:
        logger.error(f"File not found: {secret_name}")
        return None
    except Exception as e:
        logger.error(f"Error reading {secret_name}: {e}")
        return None

CLIENT_SECRET = red_secret('secret_id')
# Подумать над state нужна связка когда вызывать будет для токена от кого запрос

@app.get("/authorize")
async def start_auth():
    params = {
        "client_id": CLIENT_ID,
        "state": 8,
        "redirect_uri": "https://api.deltafeniks.kz/log",
    }
    url = "https://sip.bitrix24.kz/oauth/authorize/"
    redirect_url = requests.Request('GET', url, params=params).prepare().url
    return RedirectResponse(url=redirect_url)


@app.get("/log")
async def authorize(request: Request, db: Session = Depends(get_db)):
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state")
    domain = params.get("domain")
    member_id = params.get("member_id")
    scope = params.get("scope")
    server_domain = params.get("server_domain")

    if code:
        # Здесь можно сразу обменять code на access_token
        #return {"code": code, "state": state, "domain": domain, "member_id": member_id, "scope": scope, "server_domain": server_domain}
        # Пример запроса на получение access_token
        token_url = "https://oauth.bitrix.info/oauth/token/"
        data = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
        }
        response = requests.post(token_url, data=data)
        try:
            token_data = response.json()
        except Exception:
            token_data = {"error": "Invalid response", "text": response.text}

        bot = db.query(MyBot).order_by(MyBot.bot_id.desc()).first()
        if not bot:
            data_post = {
                'CODE' : bot.code,
                'auth': token_data.get('access_token', ''),
                'TYPE': bot.type,
                'OPENLINE': bot.openline,
                'auth': token_data.get('access_token', ''),
                'EVENT_HANDLER': 'https://api.deltafeniks.kz/bot/event',
                'EVENT_WELCOME_MESSAGE': 'https://api.deltafeniks.kz/bot/welcome',
                'PROPERTIES': {
                    'NAME': bot.name,
                    'COLOR': bot.color,
                    'EMAIL': bot.email,
                    'PERSONAL_BIRTHDAY': str(bot.personal_birthday) if bot.personal_birthday else None,
                    'WORK_POSITION': bot.work_position,
                    'PERSONAL_WWW': bot.personal_www,
                    'PERSONAL_PHOTO': encoded_string,  # Используем закодированную строку изображения
                }, 

            }
        
            result = requests.post("https://sip.bitrix24.kz/rest/imbot.register", json=data_post)
            if isinstance(result, dict):
                bot_id = result.get('BOT_ID')
            elif isinstance(result, int):
                bot_id = result
            else:
                bot_id = None
            bot.bitrix_bot_id = bot_id.json().get('result', {}).get('BOT_ID')
            db.commit()
            db.refresh(bot)
            return bot_id.json()

    else:
        return {"error": "No code received"}


# Чать для работы с базой и предсказаниями

@app.post("/create/bot")
async def create_bot(data = Body(), db: Session = Depends(get_db)):
    my_bot = MyBot(
        bitrix_bot_id=data.get("bitrix_bot_id"),
        code=data.get("code", "NewBot"),
        type=data.get("type", "O"),
        openline=data.get("openline", "Y"),
        name=data.get("name", "NewBot"),
        color=data.get("color", "GREEN"),
        email=data.get("email"),
        personal_birthday=data.get("personal_birthday", "2025-06-19"),
        work_position=data.get("work_position", "Тестриовщик API"),
        personal_www=data.get("personal_www", "https://api.deltafeniks.kz/info"))
    db.add(my_bot)
    db.commit()
    db.refresh(my_bot)
    return {"message": "Bot created successfully", "bot_id": my_bot.bot_id}



 # делаем обработчик для обработчик события отправки сообщения чат-боту.

# @app.post("/bot/event")
# async def bot_event_handler(event: dict, db: Session = Depends(get_db)):
#     # Здесь вы можете обработать событие, полученное от чат-бота
#     # Например, сохранить его в базе данных или выполнить какие-то действия
#     logger.warning(f"Received event: {event}")
#     return {"message": "Event received", "event": event}


# доделать токен вопрос с их обновлениями когда истекут, думаю лучше проверять самому чем получать
#ошибку при запросе
@app.post("/bot/event")
async def bot_event_handler(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    event_data = dict(form)
    expires_in = int(event_data.get('auth[expires_in]', 3600))  # 3600 по умолчанию, если нет значения
    expires_at = datetime.now() + timedelta(seconds=expires_in)
    user_check = db.query(Users).filter(Users.bitrix24_id == event_data.get('data[USER][ID]')).first()
    if not user_check:
        user_check = Users(
            bitrix24_id=event_data.get('data[USER][ID]'),
            name=event_data.get('data[USER][FIRST_NAME]'),
            last_active= datetime.now(),)
        db.add(user_check)
        db.commit()
        db.refresh(user_check)
        logger.warning(f"New user added: {user_check.name} with ID {user_check.bitrix24_id}")
    auth_token =  AuthTokens(
        user_id=user_check.user_id,
        access_token=event_data.get('auth[access_token]'),
        expires_at=expires_at,
        refresh_token=event_data.get('auth[refresh_token]'),
        score=event_data.get('auth[scope]'),
    )
    db.add(auth_token)
    db.commit()
    db.refresh(auth_token)

    # добавляю значение на пустой запрос
    raw_message = event_data.get('data[PARAMS][MESSAGE]')

    if not raw_message:
        logger.warning(f"Empty message received from Bitrix event: {event_data}")
        return {"error": "Empty message", "event": event_data}
    user_request = UserRequests(
        user_id=user_check.user_id,
        raw_message=raw_message,
    )
    db.add(user_request)
    db.commit()
    db.refresh(user_request)

    check = check_message(user_request.raw_message)
    logger.warning(f"Message check result: {check} for user {user_check.name} with ID {user_check.bitrix24_id}")
    if check and event_data.get('data[USER][IS_BOT]') == 'N':
        # Если сообщение является приветствием, то просто возвращаем ответ
        # и не запускаем дальнейшую обработку
        # нужно создать чат и далее туда уже переадрессовать диало
    
        message = random.choice(HELLO_MESSAGES)
        data = {'BOT_ID': 460,
                'MESSAGE': message,
                'DIALOG_ID': event_data.get('data[PARAMS][DIALOG_ID]'),
                'auth': auth_token.access_token}
        response = requests.post("https://sip.bitrix24.kz/rest/imbot.message.add", json=data)
        logger.warning(f"Response from imbot.message.add: {response.text}")
        logger.warning(f"User {user_check.name} sent a greeting message: {user_request.raw_message}")
        return {"message": "Greeting message received", "user": user_check.name}

    predict_id = predict(user_request.request_id, db)
    start_to_process(predict_id, db, event_data.get('data[PARAMS][CHAT_ID]'))
    logger.warning(f"New user request added: {user_request.raw_message} for user {user_check.name}")
    logger.warning(f"auth token: {auth_token.access_token} expires at {auth_token.expires_at}")
    logger.warning(f"message: Event received, event: {event_data}")
    #logger.warning(f"Category predicted: {category_predict} for request ID {user_request.request_id}")
    return {"message": "Event received", "event": event_data}
