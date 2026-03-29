from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
from pybo.model import db, User

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


@bp.route('/result')
def result_page():
    if 'user_id' not in session:
        return redirect(url_for('main.main_page'))

    # 여기서 DB에서 분석 결과를 가져오거나 세션에서 꺼내서 전달
    return render_template('result.html')