from flask import Flask
import oracledb
from .config import Config
from .model import db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    try:
        oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")
    except Exception as e:
        # 이미 초기화된 경우 에러가 날 수 있으므로 예외처리
        print(f"Oracle Client Status: {e}")

    # Flask-SQLAlchemy 초기화
    db.init_app(app)

    with app.app_context():
        # 앱 시작 시 테이블이 없으면 자동 생성 (오라클 연동 확인용)
        db.create_all()

    # 블루프린트 등록
    from .views import main_views
    app.register_blueprint(main_views.bp)

    return app