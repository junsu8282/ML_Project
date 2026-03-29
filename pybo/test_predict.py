import pickle
import numpy as np
import pandas as pd

# ==============================
# 1. 모델 애셋 로드 (통합 pkl 사용)
# ==============================
# 아까 save_model_assets 함수로 만든 pkl 파일을 불러옵니다.
with open("model/nutrition_model_assets.pkl", "rb") as f:
    assets = pickle.load(f)

scaler = assets["scaler"]
pca = assets["pca"]
gmm = assets["gmm"]
features = assets["features"]  # ["CARB_RATIO", "PROT_RATIO", "N_NA", "N_SUGAR", "HE_BMI", "AGE", "PA_AEROBIC"]
target_pca = assets["target_pca"]
threshold = assets["threshold"]

print("✅ GMM & PCA 모델 로드 완료")

# ==============================
# 2. 테스트 데이터 세트
# ==============================
test_samples = [
    {
        "DESC": "1. G1 (고단백 헬창) - 30대, 고단백, 운동 필수",
        "N_CHO": 200, "N_PROT": 120, "N_FAT": 50, "N_NA": 2700, "N_SUGAR": 35,
        "HE_BMI": 26.0, "AGE": 34, "PA_AEROBIC": 1.0
    },
    {
        "DESC": "2. Normal (권장 준수) - 40대, 균형 잡힌 모범생",
        "N_CHO": 300, "N_PROT": 80, "N_FAT": 60, "N_NA": 2500, "N_SUGAR": 45,
        "HE_BMI": 22.5, "AGE": 45, "PA_AEROBIC": 1.0
    },
    {
        "DESC": "3. G2 (액티브 시니어) - 50대, 운동은 하나 짜게 먹음",
        "N_CHO": 350, "N_PROT": 85, "N_FAT": 75, "N_NA": 4300, "N_SUGAR": 80,
        "HE_BMI": 25.0, "AGE": 55, "PA_AEROBIC": 1.0
    },
    {
        "DESC": "4. G0 (중장년 비활동) - 50대, 운동 안함, 식단 불균형",
        "N_CHO": 330, "N_PROT": 80, "N_FAT": 70, "N_NA": 3200, "N_SUGAR": 55,
        "HE_BMI": 24.5, "AGE": 57, "PA_AEROBIC": 0.0
    }
]

# ==============================
# 3. 추론 및 확률 계산 (Inference Logic)
# ==============================
for sample in test_samples:
    print(f"\n🚀 테스트 대상: {sample['DESC']}")

    df = pd.DataFrame([sample])
    total_kcal = (df["N_CHO"] * 4) + (df["N_PROT"] * 4) + (df["N_FAT"] * 9)
    df["CARB_RATIO"] = (df["N_CHO"] * 4) / total_kcal
    df["PROT_RATIO"] = (df["N_PROT"] * 4) / total_kcal

    X_input = df[features]
    X_scaled = scaler.transform(X_input)
    X_pca = pca.transform(X_scaled)

    # 1. Target(Normal)과의 거리 계산
    dist = np.linalg.norm(X_pca - target_pca, axis=1)[0]

    # 2. GMM 각 군집별 확률 계산 (Normal 여부 상관없이 무조건 수행)
    gmm_probs = gmm.predict_proba(X_pca)[0]  # [G0%, G1%, G2%]

    # ==========================================================
    # [수정된 로직] Normal을 포함한 4개 그룹 통합 확률 계산
    # ==========================================================
    # 거리(dist)가 작을수록 Normal 성향 점수가 높게 나오도록 지수 함수 사용
    # threshold 지점에서 Normal 성향이 적절히 낮아지도록 설계함
    normal_score = np.exp(-dist / (threshold * 0.5))

    # 나머지 GMM 성향들은 (1 - Normal성향) 내에서 비중을 나눔
    # 이렇게 해야 전체 합이 100%가 됩니다.
    rem_weight = 1 - (normal_score if normal_score < 0.9 else 0.9)  # 최소 10%는 성향 노출
    final_g0 = gmm_probs[0] * rem_weight
    final_g1 = gmm_probs[1] * rem_weight
    final_g2 = gmm_probs[2] * rem_weight

    # 다시 한번 소프트맥스처럼 합이 1이 되도록 정규화
    total = normal_score + final_g0 + final_g1 + final_g2

    prob_normal = (normal_score / total)
    prob_g0 = (final_g0 / total)
    prob_g1 = (final_g1 / total)
    prob_g2 = (final_g2 / total)

    # 3. 결과 출력 (이제 IF문 없이 모든 성향을 다 보여줌)
    print(f"   - [종합 진단 결과]")
    print(f"     * Normal(권장준수) : {prob_normal:>6.1%}")
    print(f"     * G0(중장년비활동) : {prob_g0:>6.1%}")
    print(f"     * G1(고단백헬창)  : {prob_g1:>6.1%}")
    print(f"     * G2(액티브시니어) : {prob_g2:>6.1%}")

    # 대표 페르소나 결정
    final_list = [prob_normal, prob_g0, prob_g1, prob_g2]
    names = ["Normal", "G0(중장년 비활동)", "G1(고단백 헬창)", "G2(액티브 시니어)"]
    best_idx = np.argmax(final_list)

    print(f"   => 최종 판정: {names[best_idx]} (거리: {dist:.3f})")

print("\n" + "=" * 50 + "\n검증 종료")