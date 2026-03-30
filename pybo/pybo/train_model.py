import pandas as pd
import numpy as np
import os
import oracledb
import plotly.graph_objects as go
import pickle
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score  # [복구] 실루엣 점수
from sklearn.decomposition import PCA

# [Step 0] DB 설정 (Thick 모드 유지 👊)
try:
    oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")
except Exception as e:
    print(f"Oracle Client 확인 필요: {e}")

# 계정명 대문자(MYML)가 안전합니다.
engine = create_engine("oracle+oracledb://MYML:1234@localhost:1521/xe")

def load_data():
    # 🚨 수정 포인트: 우리가 새로 만든 FEATURES 테이블에서 데이터를 가져옵니다.
    # 이미 전처리 과정에서 N_EN 등을 필터링해서 넣었으므로 쿼리가 간결해집니다.
    query = "SELECT * FROM USER_HEALTH_FEATURES"
    df = pd.read_sql(query, con=engine)
    df.columns = df.columns.str.upper()
    print(f"\n[Step 1] 새 테이블 데이터 로드 완료: {len(df)}건")
    return df


def preprocess_refined(df):
    df_work = df.copy()

    # 1. 탄/단/지 기반 칼로리 직접 계산 (FEATURES 테이블의 컬럼 사용)
    # 11g에서 가져온 데이터이므로 컬럼명이 정확히 일치하는지 확인 👊
    df_work["CALC_KCAL"] = (df_work["N_CHO"] * 4) + (df_work["N_PROT"] * 4) + (df_work["N_FAT"] * 9)

    # 2. 칼로리 대비 영양소 비율 산출
    df_work = df_work[df_work["CALC_KCAL"] > 0].copy()
    df_work["CARB_RATIO"] = (df_work["N_CHO"] * 4) / df_work["CALC_KCAL"]
    df_work["PROT_RATIO"] = (df_work["N_PROT"] * 4) / df_work["CALC_KCAL"]

    # 3. 학습 피처 정의 (FEATURES 테이블 기반)
    # N_NA, N_SUGAR, HE_BMI 등은 이미 이관할 때 정제했습니다.
    features = ["CARB_RATIO", "PROT_RATIO", "N_NA", "N_SUGAR", "HE_BMI", "AGE", "PA_AEROBIC"]

    # 결측치 제거
    df_clean = df_work.dropna(subset=features + ['SEX']).copy()

    # 4. 이상치 처리 (FEATURES 테이블 데이터 특성에 맞춰 범위 조정 가능)
    for feat in ["N_NA", "N_SUGAR", "HE_BMI"]:
        q1, q3 = df_clean[feat].quantile([0.01, 0.99])
        df_clean = df_clean[(df_clean[feat] >= q1) & (df_clean[feat] <= q3)]

    # 5. 스케일링
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_clean[features])

    # 6. 성별에 따른 타겟 포인트 (남성=1, 여성=2 기준)
    t_male = [0.55, 0.15, 2000, 50, 23.0, 45, 1.0]
    t_female = [0.55, 0.15, 2000, 50, 21.5, 45, 1.0]

    t_male_scaled = scaler.transform(pd.DataFrame([t_male], columns=features))[0]
    t_female_scaled = scaler.transform(pd.DataFrame([t_female], columns=features))[0]

    user_targets_scaled = np.where(
        df_clean[['SEX']].values == 1,
        t_male_scaled,
        t_female_scaled
    )

    print(f"[Step 2] 전처리 완료 (대상: {len(df_clean)}건)")
    return x_scaled, user_targets_scaled, features, df_clean, scaler


def train_analyze_gmm(df, x_scaled, target_scaled):
    pca_model = PCA(n_components=5)
    x_pca = pca_model.fit_transform(x_scaled)
    target_pca = pca_model.transform(target_scaled)

    dists = np.linalg.norm(x_pca - target_pca, axis=1)
    threshold = np.mean(dists) - (0.5 * np.std(dists))
    mask_normal = dists <= threshold

    df_normal = df[mask_normal].copy()
    df_features = df[~mask_normal].copy()
    x_pca_features = x_pca[~mask_normal]

    print(f"\n[임계값 설정] Distance Threshold: {threshold:.4f}")
    print(f"[Normal 선발 결과] 전체 중 {len(df_normal)}명 선발")

    # [★] GMM 최적화 검증 로그 (BIC & 실루엣 점수 복구)
    print("\n" + "=" * 60 + "\n [GMM Clustering 분석 로그] \n" + "=" * 60)
    for n in range(2, 6):
        test_gmm = GaussianMixture(n_components=n, random_state=42, n_init=10).fit(x_pca_features)
        test_labels = test_gmm.predict(x_pca_features)

        bic_score = test_gmm.bic(x_pca_features)
        sil_score = silhouette_score(x_pca_features, test_labels)
        print(f"Clusters {n} | BIC: {bic_score:.2f} | Silhouette: {sil_score:.4f}")

    # 최종 모델 학습 (N=3)
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
    labels = gmm.fit_predict(x_pca_features)

    print("-" * 60)
    print(f"==> Selected N=3 | Final BIC: {gmm.bic(x_pca_features):.2f}")

    cluster_names = {
        -1: "권장 섭취 준수군 (Normal)",
        0: "중장년 비활동 식단 불균형군 (G0)",
        1: "고단백 근성장 루틴형 (G1)",
        2: "액티브 시니어 고나트륨군 (G2)"
    }

    df_features["CLUSTER_ID"] = labels
    df_normal["CLUSTER_ID"] = -1
    df_normal["CLUSTER_NAME"] = cluster_names[-1]
    df_features["CLUSTER_NAME"] = df_features["CLUSTER_ID"].map(cluster_names)
    df_final = pd.concat([df_normal, df_features]).sort_index()

    return df_final, x_pca, target_pca, pca_model, gmm, threshold


def visualize_persona(df_final, x_pca, target_pca, gmm_model, n_components):
    # (Plotly 3D 시각화 로직 100% 보존)
    fig = go.Figure()
    color_map = {-1: '#2ECC71', 0: '#E74C3C', 1: '#F1C40F', 2: '#3498DB'}

    for cid in sorted(df_final["CLUSTER_ID"].unique()):
        mask = df_final["CLUSTER_ID"] == cid
        fig.add_trace(go.Scatter3d(
            x=x_pca[mask, 0], y=x_pca[mask, 1], z=x_pca[mask, 2],
            mode='markers',
            marker=dict(size=2 if cid == -1 else 3, color=color_map.get(cid, 'black'),
                        opacity=0.3 if cid == -1 else 0.8),
            name=df_final[mask]["CLUSTER_NAME"].iloc[0]
        ))

    fig.add_trace(go.Scatter3d(
        x=[target_pca[0, 0]], y=[target_pca[0, 1]], z=[target_pca[0, 2]],
        mode='markers+text', marker=dict(size=8, color='black', symbol='diamond'),
        name="[권장 가이드라인 지점]", text=["TARGET"]
    ))

    # 가우시안 타원체 시각화
    for i in range(n_components):
        center, cov = gmm_model.means_[i][:3], gmm_model.covariances_[i][:3, :3]
        vals, vecs = np.linalg.eigh(cov)
        r = np.sqrt(7.81)
        theta, phi = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
        x, y, z = (r * np.cos(theta) * np.sin(phi)).flatten(), (r * np.sin(theta) * np.sin(phi)).flatten(), (
                    r * np.cos(phi)).flatten()
        transformed = np.dot(vecs * np.sqrt(vals), np.array([x, y, z]))
        fig.add_trace(go.Surface(
            x=(transformed[0, :] + center[0]).reshape(20, 20),
            y=(transformed[1, :] + center[1]).reshape(20, 20),
            z=(transformed[2, :] + center[2]).reshape(20, 20),
            opacity=0.15, showscale=False, colorscale=[[0, color_map[i]], [1, color_map[i]]], hoverinfo='skip'
        ))

    fig.update_layout(title=f"Target-based Persona Mapping (N={n_components})")
    os.makedirs("plots", exist_ok=True)
    fig.write_html("plots/persona_target_report.html")


def save_model_assets(scaler, pca_model, gmm_model, features, target_pca, threshold):
    model_data = {
        "scaler": scaler, "pca": pca_model, "gmm": gmm_model,
        "features": features, "target_pca": target_pca, "threshold": threshold
    }
    with open("ml_model/nutrition_gmm_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("\n[성공] 'nutrition_gmm_model.pkl' 저장 완료")


if __name__ == "__main__":
    # 디렉토리 생성 (에러 방지)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("ml_model", exist_ok=True)

    df_raw = load_data()
    x_scaled, target_scaled, feats, df_clean, scaler = preprocess_refined(df_raw)

    # 분석 시작
    df_final, x_pca, target_pca, pca_model, gmm_model, threshold = train_analyze_gmm(df_clean, x_scaled, target_scaled)

    # 시각화 실행
    visualize_persona(df_final, x_pca, target_pca, gmm_model, n_components=3)

    # [Step 4] 최종 리포트 출력 및 DB 저장 🚀
    print("\n" + "=" * 80)
    print(f" [최종 페르소나별 평균 지표 리포트] (Target-based Normal)")
    print("=" * 80)

    # 지표 계산
    report_df = df_final.groupby("CLUSTER_NAME")[feats].mean().round(3)
    print(report_df.to_string())
    print("-" * 80)

    counts = df_final['CLUSTER_NAME'].value_counts()
    for name, cnt in counts.items():
        print(f"{name:<35} | {cnt:>5}명 ({cnt / len(df_final) * 100:>5.1f}%)")
    print("=" * 80)

    # 1. DB 저장 (USER_HEALTH_CLUSTER) 👊
    # 이제 DB의 컬럼명과 DataFrame의 컬럼명이 일치하므로 바로 밀어넣습니다.
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE USER_HEALTH_CLUSTER"))
        conn.commit()

    # 저장할 컬럼 리스트 (DB 테이블 구조와 1:1 매칭 확인)
    keep_cols = ["USER_ID", "AGE", "HE_BMI", "CARB_RATIO", "PROT_RATIO",
                 "N_NA", "N_SUGAR", "PA_AEROBIC", "CLUSTER_ID", "CLUSTER_NAME"]

    try:
        df_final[keep_cols].to_sql(
            name="user_health_cluster",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000
        )
        print(f"\n✅ USER_HEALTH_CLUSTER 테이블에 {len(df_final)}건 저장 완료!")
    except Exception as e:
        print(f"\n❌ DB 저장 중 오류 발생: {e}")

    # 2. 모델 자산 저장 (.pkl)
    save_model_assets(scaler, pca_model, gmm_model, feats, target_pca, threshold)

    print("\n✨ 모든 작업이 끝났습니다. 이제 Flask에서 이 데이터를 불러오기만 하면 됩니다! 👊")