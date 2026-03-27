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
        "DESC": "A. Normal과 G0 사이 (식단은 괜찮은데 운동 안 하는 중장년)",
        # 영양소는 Normal에 가깝지만(탄수 58%, 단백 15%), 나이가 많고 운동이 0.0
        "N_CHO": 290, "N_PROT": 75, "N_FAT": 55, "N_NA": 2100, "N_SUGAR": 50,
        "HE_BMI": 23.5, "AGE": 52, "PA_AEROBIC": 0.0
    },
    {
        "DESC": "B. G1(헬창)과 G2(시니어) 사이 (운동 열심히 하는 40대 과도기)",
        # 단백질도 챙기고(19%) 운동도 하지만, 나트륨이 높고 나이가 40대 중반
        "N_CHO": 250, "N_PROT": 125, "N_FAT": 60, "N_NA": 4500, "N_SUGAR": 45,
        "HE_BMI": 25.0, "AGE": 40, "PA_AEROBIC": 1.0
    },
    {
        "DESC": "C. G0(불균형)와 G2(시니어) 사이 (식단은 최악인데 운동 깔짝 하는 분)",
        # 나트륨/당류 폭발, 나이 56세, 근데 유산소 운동은 조금 함(0.5)
        "N_CHO": 400, "N_PROT": 65, "N_FAT": 70, "N_NA": 4200, "N_SUGAR": 80,
        "HE_BMI": 27.0, "AGE": 56, "PA_AEROBIC": 0.5
    }
]

# ==============================
# 3. 추론 및 확률 계산 (Inference Logic)
# ==============================
for sample in test_samples:
    print(f"\n🚀 테스트 대상: {sample['DESC']}")

    # 데이터프레임 변환
    df = pd.DataFrame([sample])

    # [🔥 중요] 학습 때와 동일한 Feature Engineering
    total_kcal = (df["N_CHO"] * 4) + (df["N_PROT"] * 4) + (df["N_FAT"] * 9)
    df["CARB_RATIO"] = (df["N_CHO"] * 4) / total_kcal
    df["PROT_RATIO"] = (df["N_PROT"] * 4) / total_kcal

    # 7개 피처만 추출 및 스케일링
    X_input = df[features]
    X_scaled = scaler.transform(X_input)

    # PCA 변환
    X_pca = pca.transform(X_scaled)

    # [Step 1] Normal 체크 (유클리드 거리)
    dist = np.linalg.norm(X_pca - target_pca, axis=1)[0]

    if dist <= threshold:
        print(f"   - 결과: [권장 섭취 준수군 (Normal)]")
        print(f"   - 타겟 거리: {dist:.4f} (기준선: {threshold:.4f} 이하로 합격)")
    else:
        # [Step 2] Normal이 아니면 GMM 확률 계산
        # predict_proba는 [[prob0, prob1, prob2]] 형태의 배열을 반환함
        probs = gmm.predict_proba(X_pca)[0]

        # 확률 순서대로 이름 매칭
        cluster_map = {0: "G0(중장년 비활동)", 1: "G1(고단백 헬창)", 2: "G2(액티브 시니어)"}
        best_cluster = np.argmax(probs)  # 가장 확률 높은 인덱스

        print(f"   - 결과: [{cluster_map[best_cluster]}]")
        print(f"   - 타겟 거리: {dist:.4f} (기준선: {threshold:.4f} 초과로 불합격)")
        print(f"   - 상세 확률: G0:{probs[0]:.1%}, G1:{probs[1]:.1%}, G2:{probs[2]:.1%}")

print("\n" + "=" * 50 + "\n검증 종료")