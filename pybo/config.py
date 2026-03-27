import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = "oracle+oracledb://myml:1234@localhost:1521/xe"
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = "dev"