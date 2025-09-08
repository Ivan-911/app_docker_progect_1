import os
from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, relationship, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, \
    ForeignKey, PrimaryKeyConstraint, Enum, Date, CheckConstraint, Text, func, UniqueConstraint
from urllib.parse import quote_plus
from fastapi import FastAPI
from sqlalchemy import MetaData

# password = quote_plus("dr47lord@!2020") 
# mysql_url = f"mysql+pymysql://admin:{password}@127.0.0.1:3306/bitrix_integration?charset=utf8mb4"
mysql_url = os.getenv("DATABASE_URL", "mysql+pymysql://ivan:dr47lord@127.0.0.1:3306/my_app_db?charset=utf8mb4")

engine = create_engine(mysql_url, echo=True, future=True)

metadata = MetaData(
    naming_convention={},
    # Опционально: общий charset и collation
)

Base = declarative_base(metadata=metadata)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# class MyBot(Base):
#     __tablename__ = 'my_bot'

#     bot_id = Column(Integer, primary_key=True, autoincrement=True)  # Уникальный идентификатор бота
#     bitrix_bot_id = Column(Integer, unique=True)                   # Уникальный ID бота в Bitrix24
#     code = Column(String(50))                                      # Код бота
#     type = Column(String(2), default='O')                          # Тип (по умолчанию 'O')
#     openline = Column(String(2), default='Y')                      # Открытая линия (по умолчанию 'Y')
#     name = Column(String(20), default='NewBot')                    # Имя бота (по умолчанию 'NewBot')
#     color = Column(Enum('GREEN', 'RED', 'BLUE', 'YELLOW'), default='GREEN')  # Цвет (enum)
#     email = Column(String(100))                                    # Email (ограничение на формат реализуется отдельно)
#     personal_birthday = Column(Date, default='2025-06-19')         # Дата рождения (по умолчанию 2025-06-19)
#     work_position = Column(String(100), default='Тестриовщик API') # Должность (по умолчанию 'Тестриовщик API')
#     personal_www = Column(String(50), default='https://api.deltafeniks.kz/info') # Сайт (по умолчанию)
#     personal_gender = Column(Enum('M', 'F'), default='M')          # Пол (enum, по умолчанию 'M')

#     __table_args__ = (
#         CheckConstraint("email LIKE '%_@__%.__%'", name='check_email_format'),
#     )



    
# class UserRequests(Base):
#     __tablename__ = 'user_requests'

#     request_id = Column(Integer, primary_key=True, autoincrement=True)  # Уникальный идентификатор запроса
#     user_id = Column(Integer, nullable=False)                           # ID пользователя (обязателен)
#     raw_message = Column(Text, nullable=False)                          # Оригинальное сообщение

#     # Опционально: связь с ML предсказаниями
#     ml_predicts = relationship("MLPredict", back_populates="user_request", cascade="all, delete-orphan")


# class MLPredict(Base):
#     __tablename__ = 'ml_predict'

#     predict_id = Column(Integer, primary_key=True, autoincrement=True)  # Уникальный идентификатор предсказания
#     predict_label = Column(Text, nullable=False)                        # Метка предсказания (обязательна)
#     predict_message = Column(Text)                                      # Сообщение предсказания
#     request_id = Column(Integer, ForeignKey('user_requests.request_id', ondelete="CASCADE"))  # Внешний ключ на user_requests

#     # Опционально: связь с запросом пользователя
#     user_request = relationship("UserRequests", back_populates="ml_predicts")
#     user = relationship("Users", back_populates="ml_predicts", uselist=False)



    
# class Users(Base):
#     __tablename__ = 'users'

#     user_id = Column(Integer, primary_key=True, autoincrement=True)           # Уникальный идентификатор пользователя
#     bitrix24_id = Column(Integer, unique=True, nullable=True)                 # Уникальный Bitrix24 ID (может быть NULL)
#     email = Column(String(100), unique=True, nullable=True)                   # Email (уникальный, может быть NULL)
#     name = Column(String(100), nullable=True)                                 # Имя пользователя (может быть NULL)
#     last_active = Column(DateTime, nullable=True)                             # Дата последней активности (может быть NULL)
#     created_at = Column(DateTime, nullable=True, server_default=func.now())   # Дата создания (по умолчанию текущее время)

#     __table_args__ = (
#         UniqueConstraint('bitrix24_id', name='uq_bitrix24_id'),
#         UniqueConstraint('email', name='uq_email'),
#     )    


    
# class AuthTokens(Base):
#     __tablename__ = 'auth_tokens'

#     token_id = Column(Integer, primary_key=True, autoincrement=True)           # Уникальный идентификатор токена
#     user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)     # Внешний ключ на пользователя
#     access_token = Column(String(512), nullable=False)                         # Access token (обязателен)
#     refresh_token = Column(String(512), nullable=False)                        # Refresh token (обязателен)
#     expires_at = Column(DateTime, nullable=False)                              # Время истечения токена (обязательно)
#     score = Column(String(255), nullable=True)                                 # Дополнительное поле (может быть пустым)

#     # Связь с пользователем
#     user = relationship("Users", back_populates="auth_tokens")

class Users(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    bitrix24_id = Column(Integer, unique=True, nullable=True)
    email = Column(String(100), unique=True, nullable=True)
    name = Column(String(100), nullable=True)
    last_active = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=True, server_default=func.now())

    __table_args__ = (
        UniqueConstraint('bitrix24_id', name='uq_bitrix24_id'),
        UniqueConstraint('email', name='uq_email'),
    )

    # --- связи ---
    auth_tokens = relationship("AuthTokens", back_populates="user", cascade="all, delete-orphan")
    requests = relationship("UserRequests", back_populates="user", cascade="all, delete-orphan")


class AuthTokens(Base):
    __tablename__ = 'auth_tokens'

    token_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    access_token = Column(String(512), nullable=False)
    refresh_token = Column(String(512), nullable=True, default=None)
    expires_at = Column(DateTime, nullable=False)
    score = Column(String(255), nullable=True)

    # связь с Users
    user = relationship("Users", back_populates="auth_tokens")


class UserRequests(Base):
    __tablename__ = 'user_requests'

    request_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    visitor = Column(String(255), nullable=True)  # как в схеме
    raw_message = Column(Text, nullable=True, default=None)

    # связи
    user = relationship("Users", back_populates="requests")
    ml_predicts = relationship("MLPredict", back_populates="user_request", cascade="all, delete-orphan")


class MLPredict(Base):
    __tablename__ = 'ml_predict'

    predict_id = Column(Integer, primary_key=True, autoincrement=True)
    predict_label = Column(Text, nullable=False)
    predict_message = Column(Text, nullable=True)
    request_id = Column(Integer, ForeignKey('user_requests.request_id', ondelete="CASCADE"))

    # связь с UserRequests
    user_request = relationship("UserRequests", back_populates="ml_predicts")


class MyBot(Base):
    __tablename__ = 'my_bot'

    bot_id = Column(Integer, primary_key=True, autoincrement=True)
    bitrix_bot_id = Column(Integer, unique=True)
    code = Column(String(50))
    type = Column(String(2), default="O")
    openline = Column(String(2), default="Y")
    name = Column(String(20), default="NewBot")
    color = Column(Enum('GREEN', 'RED', 'BLUE', 'YELLOW'), default='GREEN')
    email = Column(String(100))
    personal_birthday = Column(DateTime, default="2025-06-19")
    work_position = Column(String(100), default="Тестировщик API")
    personal_www = Column(String(50), default="https://api.deltafeniks.kz/info")
    personal_gender = Column(Enum('M', 'F'), default="M")

    __table_args__ = (
        CheckConstraint("email LIKE '%_@__%.__%'", name='check_email_format'),
    )

# --- функция инициализации базы ---
def init_db():
    """Создаёт все таблицы, если их ещё нет"""
    Base.metadata.create_all(bind=engine)