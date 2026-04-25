from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
from pybo.model import db, User, UserInputData
from pybo.predict_cluster import predict_user_persona
from sqlalchemy import text

bp = Blueprint('main', __name__, url_prefix='/')


# 1. 메인 페이지
@bp.route('/')
def main_page():
    return render_template('home.html')


@bp.route('/mypage')
def mypage():
    # 세션에서 현재 로그인한 유저의 이메일 가져오기
    user_email = session.get('user_id')

    # 1. DB에서 해당 유저의 모든 분석 기록 조회 (최신순)
    records = UserInputData.query.filter_by(user_id=user_email).order_by(UserInputData.created_at.desc()).all()

    return render_template('mypage.html', history=records)


@bp.route('/api/get_result/<int:result_id>')
def get_result_by_id(result_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': '로그인 필요'}), 401

    try:
        # 최신순이 아니라, 클릭한 그 'INPUT_ID'로 조회합니다!
        query = text("SELECT * FROM USER_INPUT_DATA WHERE INPUT_ID = :result_id")
        row = db.session.execute(query, {"result_id": result_id}).fetchone()

        if row:
            res = {k.lower(): v for k, v in row._mapping.items()}

            # AI 모델 재예측 (기존 로직 유지)
            user_input_dict = {
                'N_CHO': res['n_cho'], 'N_PROT': res['n_prot'], 'N_FAT': res['n_fat'],
                'N_NA': res['n_na'], 'N_SUGAR': res['n_sugar'],
                'HE_BMI': res['he_bmi'], 'AGE': res['age'], 'PA_AEROBIC': res['pa_aerobic']
            }
            prediction = predict_user_persona(user_input_dict)
            res.update(prediction)

            return jsonify({'status': 'success', 'data': res})
        return jsonify({'status': 'error', 'message': '기록을 찾을 수 없습니다.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 2. 로그인 경로 처리 (선택 사항)
@bp.route('/login')
def login_page():
    return redirect(url_for('main.main_page'))

# --- 로그인/회원가입  ---
@bp.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data.get('user_id')
    password = data.get('password')

    user = User.query.filter_by(user_id=user_id, password=password).first()

    if user:
        session['user_id'] = user.user_id
        session['user_name'] = user.user_name

        # JS에서 window.location.href = "/info"로 이동하게 되므로 status만 넘깁니다.
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "아이디 또는 비밀번호가 틀렸습니다."}), 401


@bp.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user_name = data.get('user_name')
    user_id = data.get('user_id')
    password = data.get('password')

    if User.query.filter_by(user_id=user_id).first():
        return jsonify({"status": "error", "message": "이미 존재하는 아이디입니다."}), 400

    new_user = User(user_name=user_name, user_id=user_id, password=password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status": "success", "message": "회원가입이 완료되었습니다!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"DB 오류: {str(e)}"}), 500


@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('main.main_page'))


@bp.route('/nutrient')
def nutrient_page():
    # 1. 로그인 체크 (세션에 user_id가 없으면 로그인 페이지로)
    if 'user_id' not in session:
        return redirect(url_for('main.main_page'))

    return render_template('nutrient_input.html')


@bp.route('/info')
def info_page():
    # 세션 체크: 로그인 안 한 사용자가 주소창에 /info 쳐서 들어오는 것 방지
    if 'user_id' not in session:
        return redirect(url_for('main.main_page'))

    # 신체 정보 입력 페이지 보여주기
    return render_template('info_input.html')


@bp.route('/api/simulate_analysis', methods=['POST'])
def simulate_analysis():
    try:
        data = request.get_json()

        # 1. JS payload 키값에 맞춰 데이터 추출
        age = int(data.get('age', 30))
        gender = 1 if session.get('gender') == 'male' else 2
        height = float(data.get('height', 175)) / 100
        weight = float(data.get('weight', 70))
        he_bmi = round(weight / (height ** 2), 2)

        # JS에서 PA_AEROBIC(0 또는 1)로 보내므로 바로 받음
        pa_aerobic = int(data.get('PA_AEROBIC', 0))

        # 영양소 키값 매핑 (N_CHO, N_PROT 등으로 통일)
        n_cho = float(data.get('N_CHO', 0))
        n_prot = float(data.get('N_PROT', 0))
        n_fat = float(data.get('N_FAT', 0))
        n_sugar = float(data.get('N_SUGAR', 0))
        n_na = float(data.get('N_NA', 0))

        # 2. 비율 계산 (차트용)
        total_kcal = (n_cho * 4) + (n_prot * 4) + (n_fat * 9)
        c_ratio = (n_cho * 4) / total_kcal if total_kcal > 0 else 0
        p_ratio = (n_prot * 4) / total_kcal if total_kcal > 0 else 0
        f_ratio = (n_fat * 9) / total_kcal if total_kcal > 0 else 0

        # 3. AI 모델 예측
        user_input_for_ml = {
            'N_CHO': n_cho, 'N_PROT': n_prot, 'N_FAT': n_fat,
            'N_NA': n_na, 'N_SUGAR': n_sugar,
            'HE_BMI': he_bmi, 'AGE': age, 'SEX': gender, 'PA_AEROBIC': pa_aerobic
        }
        analysis_result = predict_user_persona(user_input_for_ml)

        return jsonify({
            'status': 'success',
            'data': {
                'cluster_name': analysis_result['cluster_name'],
                'cluster_id': analysis_result['cluster_id'],
                'probabilities': analysis_result['probabilities'],
                'carb_ratio': c_ratio,
                'prot_ratio': p_ratio,
                'fat_ratio': f_ratio,
                'n_cho': n_cho, 'n_prot': n_prot, 'n_fat': n_fat,
                'n_sugar': n_sugar, 'n_na': n_na,
                'sex': gender
            }
        })
    except Exception as e:
        print(f"❌ 시뮬레이션 오류: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@bp.route('/api/save_analysis', methods=['POST'])
def save_analysis():
    try:
        data = request.get_json()

        # 1. 입력 데이터 추출
        age = int(data.get('age'))
        gender = 1 if data.get('gender') == 'male' else 2
        height = float(data.get('height')) / 100
        weight = float(data.get('weight'))

        he_bmi = round(weight / (height ** 2), 2)
        activity = float(data.get('activity'))
        pa_aerobic = 1 if activity >= 1.55 else 0

        # 영양소 (원본 수치)
        n_cho = float(data.get('carbs'))
        n_prot = float(data.get('protein'))
        n_fat = float(data.get('fat'))
        n_sugar = float(data.get('sugar'))
        n_na = float(data.get('sodium'))

        # 2. 탄/단/지 비율만 계산 (차트용) 👊
        total_kcal = (n_cho * 4) + (n_prot * 4) + (n_fat * 9)
        c_ratio = round((n_cho * 4) / total_kcal, 4) if total_kcal > 0 else 0
        p_ratio = round((n_prot * 4) / total_kcal, 4) if total_kcal > 0 else 0
        f_ratio = round((n_fat * 9) / total_kcal, 4) if total_kcal > 0 else 0

        # 3. AI 모델 예측
        user_input_for_ml = {
            'N_CHO': n_cho, 'N_PROT': n_prot, 'N_FAT': n_fat,
            'N_NA': n_na, 'N_SUGAR': n_sugar,
            'HE_BMI': he_bmi, 'AGE': age, 'SEX': gender, 'PA_AEROBIC': pa_aerobic
        }
        analysis_result = predict_user_persona(user_input_for_ml)

        # 4. DB 저장 (에러 났던 sugar_ratio, na_ratio 삭제!) 🚀
        save_query = text("""
            INSERT INTO USER_INPUT_DATA (
                INPUT_ID, USER_ID, N_CHO, N_PROT, N_FAT, N_NA, N_SUGAR,
                HE_BMI, AGE, SEX, PA_AEROBIC, PREDICTED_CLUSTER,
                CARB_RATIO, PROT_RATIO, FAT_RATIO
            )
            VALUES (
                INPUT_ID_SEQ.NEXTVAL, :user_id, :n_cho, :n_prot, :n_fat, :n_na, :n_sugar,
                :he_bmi, :age, :sex, :pa_aerobic, :predicted_cluster,
                :c_ratio, :p_ratio, :f_ratio
            )
        """)

        db.session.execute(save_query, {
            'user_id': session.get('user_id', 'GUEST'),
            'n_cho': n_cho, 'n_prot': n_prot, 'n_fat': n_fat, 'n_na': n_na, 'n_sugar': n_sugar,
            'he_bmi': he_bmi, 'age': age, 'sex': gender, 'pa_aerobic': pa_aerobic,
            'predicted_cluster': analysis_result['cluster_name'],
            'c_ratio': c_ratio, 'p_ratio': p_ratio, 'f_ratio': f_ratio
        })
        db.session.commit()

        return jsonify({'status': 'success', 'redirect_url': url_for('main.result_page')})

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 1. 결과 페이지 진입 (껍데기만 로드)
@bp.route('/result/<int:result_id>')
@bp.route('/result')
def result_page(result_id=None):
    return render_template('result.html', result_id=result_id)


# 2. AJAX 요청을 처리하는 데이터 전용 API
@bp.route('/api/get_latest_result')
def get_latest_result():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': '로그인 세션이 만료되었습니다.'}), 401

    try:
        # DB에서 데이터 가져오기
        query = text("SELECT * FROM USER_INPUT_DATA WHERE USER_ID = :user_id ORDER BY INPUT_ID DESC")
        row = db.session.execute(query, {"user_id": user_id}).fetchone()

        if row:
            res = {k.lower(): v for k, v in row._mapping.items()}

            # 2. 함수에 넣을 데이터 형식 맞추기 (대문자 키값)
            user_input_dict = {
                'N_CHO': res['n_cho'],
                'N_PROT': res['n_prot'],
                'N_FAT': res['n_fat'],
                'N_NA': res['n_na'],
                'N_SUGAR': res['n_sugar'],
                'HE_BMI': res['he_bmi'],
                'AGE': res['age'],
                'PA_AEROBIC': res['pa_aerobic']
            }

            # 3. 이미 만든 함수 호출해서 결과 받기 🚀
            prediction = predict_user_persona(user_input_dict)

            # 4. 결과 합치기
            res.update(prediction)  # cluster_id, cluster_name 등이 들어감

            return jsonify({'status': 'success', 'data': res})

        else:
            return jsonify({'status': 'error', 'message': '데이터 없음'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500