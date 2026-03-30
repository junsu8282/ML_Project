import pickle
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "nutrition_gmm_model.pkl")

def predict_user_persona(user_input):
    # [수정] 계산된 절대 경로(MODEL_PATH)를 사용합니다.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model_assets = pickle.load(f)

    scaler = model_assets["scaler"]
    pca = model_assets["pca"]
    gmm = model_assets["gmm"]
    features = model_assets["features"]
    target_pca = model_assets["target_pca"]
    threshold = model_assets["threshold"]

    # 파생 변수 계산
    calc_kcal = (user_input['N_CHO'] * 4) + (user_input['N_PROT'] * 4) + (user_input['N_FAT'] * 9)
    carb_ratio = (user_input['N_CHO'] * 4) / calc_kcal if calc_kcal > 0 else 0
    prot_ratio = (user_input['N_PROT'] * 4) / calc_kcal if calc_kcal > 0 else 0

    input_data = pd.DataFrame([{
        "CARB_RATIO": carb_ratio,
        "PROT_RATIO": prot_ratio,
        "N_NA": user_input['N_NA'],
        "N_SUGAR": user_input['N_SUGAR'],
        "HE_BMI": user_input['HE_BMI'],
        "AGE": user_input['AGE'],
        "PA_AEROBIC": user_input['PA_AEROBIC']
    }], columns=features)

    x_scaled = scaler.transform(input_data)
    x_pca = pca.transform(x_scaled)

    # Normal 판정 로직
    dist_to_target = np.linalg.norm(x_pca - target_pca)

    if dist_to_target <= threshold:
        cluster_id = -1
        cluster_name = "권장 섭취 준수군 (Normal)"
    else:
        cluster_id = gmm.predict(x_pca)[0]
        names = {0: "중장년 비활동 식단 불균형군 (G0)", 1: "고단백 근성장 루틴형 (G1)", 2: "액티브 시니어 고나트륨군 (G2)"}
        cluster_name = names.get(cluster_id, "기타 불균형군")

    return {
        "cluster_id": int(cluster_id),
        "cluster_name": cluster_name,
        "dist_score": round(float(dist_to_target), 4),
        "carb_ratio": round(carb_ratio, 2),
        "prot_ratio": round(prot_ratio, 2)
    }