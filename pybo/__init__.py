from flask import Flask
from config import *


def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SECRET_KEY"] = SECRET_KEY

    from .routes import bp
    app.register_blueprint(bp)

    return app


app = create_app()