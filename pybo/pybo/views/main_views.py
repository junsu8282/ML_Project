from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
from pybo.model import db, User
from pybo.predict_cluster import predict_user_persona
from sqlalchemy import text

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/')
def main_page():
    # 1. 이미 로그인 세션이 있다면 정보 입력 페이지로 리다이렉트
    if 'user_id' in session:
        return redirect(url_for('main.info_page'))

    # 2. 로그인 안 되어 있으면 'auth.html'을 보여줍니다. (base.html이 아닙니다!)
    # auth.html이 base.html을 상속받고 있으므로 디자인은 자동으로 따라옵니다.
    return render_template('auth.html')


# --- 아래 로그인/회원가입 로직은 그대로 유지하되, 리다이렉트 경로만 확인하세요 ---

@bp.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data.get('user_id')
    password = data.get('password')

    user = User.query.filter_by(user_id=user_id, password=password).first()

    if user:
        session['user_id'] = user_id
        # JS에서 window.location.href = "/info"로 이동하게 되므로 status만 넘깁니다.
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "아이디 또는 비밀번호가 틀렸습니다."}), 401


@bp.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user_id = data.get('user_id')
    password = data.get('password')

    if User.query.filter_by(user_id=user_id).first():
        return jsonify({"status": "error", "message": "이미 존재하는 아이디입니다."}), 400

    new_user = User(user_id=user_id, password=password)
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status": "success", "message": "회원가입이 완료되었습니다!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"DB 오류: {str(e)}"}), 500


@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('main.main_page'))


@bp.route('/nutrient')
def nutrient_page():
    # 1. 로그인 체크 (세션에 user_id가 없으면 로그인 페이지로)
    if 'user_id' not in session:
        return redirect(url_for('main.main_page'))

    # 2. (선택사항) 1단계 정보를 입력했는지 체크
    # 만약 1단계 데이터를 서버 세션에 저장했다면 여기서 체크할 수 있습니다.
    # 하지만 준수님은 localStorage(클라이언트)를 쓰시니까,
    # 여기서는 페이지를 열어주고 JS에서 체크하게 두는 게 편합니다.

    return render_template('nutrient_input.html')


@bp.route('/info')
def info_page():
    # 세션 체크: 로그인 안 한 사용자가 주소창에 /info 쳐서 들어오는 것 방지
    if 'user_id' not in session:
        return redirect(url_for('main.main_page'))

    # 신체 정보 입력 페이지 보여주기
    return render_template('info_input.html')


@bp.route('/api/save_analysis', methods=['POST'])
def save_analysis():
    try:
        data = request.get_json()

        # 1. 입력 데이터 추출 및 변환 👊
        age = int(data.get('age'))
        gender = 1 if data.get('gender') == 'male' else 2
        height = float(data.get('height')) / 100  # cm -> m 변환
        weight = float(data.get('weight'))

        # BMI 계산 (몸무게 / 키^2)
        he_bmi = round(weight / (height ** 2), 2)

        # 활동량 (PA_AEROBIC 판별: 여기선 보통 활동 1.55 이상을 1로 가정하거나 그대로 사용)
        activity = float(data.get('activity'))
        pa_aerobic = 1 if activity >= 1.55 else 0

        # 영양소 데이터
        n_cho = float(data.get('carbs'))
        n_prot = float(data.get('protein'))
        n_fat = float(data.get('fat'))
        n_sugar = float(data.get('sugar'))
        n_na = float(data.get('sodium'))

        # 2. AI 모델에 넣을 형식으로 패킹
        user_input_for_ml = {
            'N_CHO': n_cho, 'N_PROT': n_prot, 'N_FAT': n_fat,
            'N_NA': n_na, 'N_SUGAR': n_sugar,
            'HE_BMI': he_bmi, 'AGE': age, 'SEX': gender, 'PA_AEROBIC': pa_aerobic
        }

        # 3. 페르소나 예측 실행! 👊
        analysis_result = predict_user_persona(user_input_for_ml)

        # 4. DB 저장 (USER_INPUT_DATA 테이블)
        # 11g 시퀀스/트리거가 작동하도록 필드 구성
        save_query = text("""
                          INSERT INTO USER_INPUT_DATA (INPUT_ID, USER_ID, N_CHO, N_PROT, N_FAT, N_NA, N_SUGAR,
                                                       HE_BMI, AGE, SEX, PA_AEROBIC, PREDICTED_CLUSTER)
                          VALUES (INPUT_ID_SEQ.NEXTVAL, :user_id, :n_cho, :n_prot, :n_fat, :n_na, :n_sugar,
                                  :he_bmi, :age, :sex, :pa_aerobic, :predicted_cluster)
                          """)

        db.session.execute(save_query, {
            'user_id': session.get('user_id', 'GUEST'),  # 로그인 기능 연동 시
            'n_cho': n_cho, 'n_prot': n_prot, 'n_fat': n_fat, 'n_na': n_na, 'n_sugar': n_sugar,
            'he_bmi': he_bmi, 'age': age, 'sex': gender, 'pa_aerobic': pa_aerobic,
            'predicted_cluster': analysis_result['cluster_name']
        })
        db.session.commit()

        # 5. 결과를 세션에 저장 (결과 페이지에서 보여주기 용도)
        session['last_analysis'] = analysis_result

        return jsonify({
            'status': 'success',
            'redirect_url': url_for('main.result_page')  # 결과 페이지 경로
        })

    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@bp.route('/result')
def result_page():
    # 세션에서 방금 분석한 결과 꺼내기 👊
    result = session.get('last_analysis')

    if not result:
        # 만약 세션에 데이터가 없다면 분석 페이지로 돌려보내기
        return redirect(url_for('main.info_page'))

        # result.html에 분석 데이터를 변수로 넘겨줍니다.
    return render_template('result.html', result=result)