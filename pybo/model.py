import pickle
import numpy as np
import pandas as pd
import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'USERS'  # 오라클 DB에 생성될 테이블 이름
    user_name = db.Column(db.String(50), nullable=False)  # 이름
    user_id = db.Column(db.String(50), primary_key=True)  # 아이디 (PK)
    password = db.Column(db.String(100), nullable=False)  # 비밀번호


class UserInputData(db.Model):
    __tablename__ = 'USER_INPUT_DATA'

    # name 속성을 이용해 DB의 실제 대문자 컬럼명과 매핑합니다.
    id = db.Column('INPUT_ID', db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column('USER_ID', db.String(50), db.ForeignKey('USERS.user_id'), nullable=False)

    age = db.Column('AGE', db.Integer)
    gender = db.Column('SEX', db.String(10))
    bmi = db.Column('HE_BMI', db.Float)
    activity = db.Column('PA_AEROBIC', db.Float)

    n_cho = db.Column('N_CHO', db.Float)
    n_prot = db.Column('N_PROT', db.Float)
    n_fat = db.Column('N_FAT', db.Float)
    n_na = db.Column('N_NA', db.Float)
    n_sugar = db.Column('N_SUGAR', db.Float)

    result_type = db.Column('PREDICTED_CLUSTER', db.String(200))
    created_at = db.Column('CREATED_AT', db.DateTime, default=db.func.now())

    user = db.relationship('User', backref=db.backref('analyses', lazy=True))


class NutritionAI:
    def __init__(self):
        # 경로 설정 (현재 파일 위치 기준)
        base_path = os.path.dirname(__file__)
        pkl_path = os.path.join(base_path, "ml_model", "nutrition_gmm_model.pkl")

        with open(pkl_path, "rb") as f:
            assets = pickle.load(f)

        self.scaler = assets["scaler"]
        self.pca = assets["pca"]
        self.gmm = assets["gmm"]
        self.features = assets["features"]
        self.target_pca = assets["target_pca"]
        self.threshold = assets["threshold"] - 0.1

    def get_persona(self, data):
        df = pd.DataFrame([data])

        # Feature Engineering
        total_kcal = (df["N_CHO"] * 4) + (df["N_PROT"] * 4) + (df["N_FAT"] * 9)
        df["CARB_RATIO"] = (df["N_CHO"] * 4) / total_kcal
        df["PROT_RATIO"] = (df["N_PROT"] * 4) / total_kcal

        X_scaled = self.scaler.transform(df[self.features])
        X_pca = self.pca.transform(X_scaled)

        # 확률 계산 로직 (우리 검증본)
        dist = np.linalg.norm(X_pca - self.target_pca, axis=1)[0]
        gmm_probs = self.gmm.predict_proba(X_pca)[0]

        n_score = np.exp(-dist / (self.threshold * 0.5))
        rem_w = 1 - (n_score if n_score < 0.9 else 0.9)

        probs = [n_score, gmm_probs[0] * rem_w, gmm_probs[1] * rem_w, gmm_probs[2] * rem_w]
        total = sum(probs)
        final_probs = [p / total for p in probs]

        names = ["Normal", "G0(비활동)", "G1(헬창)", "G2(시니어)"]
        return {
            "main": names[np.argmax(final_probs)],
            "details": {n: round(p * 100, 1) for n, p in zip(names, final_probs)},
            "dist": round(dist, 3)
        }


# 싱글톤 객체 생성
ai_engine = NutritionAI()