import os

class Config:
    SECRET_KEY = "dev_junsu_key"
    # oracledb 라이브러리 방식에 맞춘 URI
    SQLALCHEMY_DATABASE_URI = "oracle+oracledb://myml:1234@localhost:1521/xe"
    SQLALCHEMY_TRACK_MODIFICATIONS = False