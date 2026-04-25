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

    if 'height' in user_input and 'weight' in user_input:
        height_m = user_input['height'] / 100
        current_bmi = user_input['weight'] / (height_m ** 2)
    else:
        # 슬라이더 값이 없을 경우 기존 DB의 BMI 사용
        current_bmi = user_input.get('HE_BMI', 22.0)

    input_data = pd.DataFrame([{
        "CARB_RATIO": carb_ratio,
        "PROT_RATIO": prot_ratio,
        "N_NA": user_input['N_NA'],
        "N_SUGAR": user_input['N_SUGAR'],
        "HE_BMI": current_bmi,
        "AGE": user_input['AGE'],
        "PA_AEROBIC": user_input['PA_AEROBIC']
    }], columns=features)

    x_scaled = scaler.transform(input_data)
    x_pca = pca.transform(x_scaled)

    if x_pca.ndim != 2:
        x_pca = x_pca.reshape(1, -1)

    # 1. Target(Normal)과의 거리 계산
    dist_to_target = np.linalg.norm(x_pca[0] - target_pca)

    # 사용자의 원본 데이터(n_na, n_sugar)를 기준으로 판정
    is_clean_diet = (user_input['N_NA'] <= 2500 and user_input['N_SUGAR'] <= 55)

    # 식단이 좋다면 기존 threshold의 1.2배까지 정상으로 인정
    effective_threshold = threshold * (1.2 if is_clean_diet else 1.0)
    # ---------------------------------------------------------

    # 2. GMM 각 군집별 확률 계산
    gmm_probs = gmm.predict_proba(x_pca)[0]

    # 3. [통합 확률 계산]
    # 이제 고정된 threshold 대신 effective_threshold를 사용해 점수를 냅니다.
    normal_score = np.exp(-dist_to_target / (effective_threshold * 0.5))

    # 나머지 로직은 동일 (rem_weight 계산 등)
    rem_weight = 1.0 - min(normal_score, 0.95)

    raw_probs = [
        normal_score,
        gmm_probs[0] * rem_weight,
        gmm_probs[1] * rem_weight,
        gmm_probs[2] * rem_weight
    ]
    total = sum(raw_probs)
    final_prob = [p / total for p in raw_probs]

    # [수정 포인트] 예외 처리 (2030 클린 식단) 로직을 먼저 수행
    is_young_and_clean = (user_input['N_NA'] <= 2500 and
                          user_input['N_SUGAR'] <= 55 and
                          user_input['AGE'] < 40)

    if is_young_and_clean:
        # 정상 확률을 70%로 올리는 보정 수행
        if final_prob[0] < 0.7:
            old_normal = final_prob[0]
            final_prob[0] = 0.7
            remaining = 0.3
            other_sum = sum(final_prob[1:])
            if other_sum > 0:
                for i in range(1, 4):
                    final_prob[i] = (final_prob[i] / other_sum) * remaining
            else:
                final_prob = [0.7, 0.1, 0.1, 0.1]

    # [중요] 모든 보정이 끝난 "최종 확률"을 바탕으로 1등(best_idx)을 뽑아야 함
    best_idx = int(np.argmax(final_prob))

    display_names = ["균형 잡힌 식단의 정석", "에너지 대사 주의 식단", "활기찬 에너자이저 식단", "탄탄한 기초 안정 식단"]
    cluster_name = display_names[best_idx]  # 이제 여기서 0번(정석)을 제대로 가져옵니다.

    cluster_id = -1 if best_idx == 0 else (best_idx - 1)

    return {
        "cluster_id": int(cluster_id),
        "cluster_name": cluster_name,
        "probabilities": [round(p, 4) for p in final_prob],
        "cluster_names": display_names,
        "dist_score": round(float(dist_to_target), 4),
        "carb_ratio": round(carb_ratio, 2),
        "prot_ratio": round(prot_ratio, 2),
        "bmi": round(current_bmi, 1)
    }