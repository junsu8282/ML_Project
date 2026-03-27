import pickle

# 서버 시작할 때 한 번만 로드
model = pickle.load(open("pybo/model.pkl", "rb"))
scaler = pickle.load(open("pybo/scaler.pkl", "rb"))