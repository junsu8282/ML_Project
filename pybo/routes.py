from flask import Blueprint, render_template, request
import pickle
import numpy as np

bp = Blueprint("main", __name__)

# 모델 로드
with open("pybo/model/kmeans.pkl", "rb") as f:
    model = pickle.load(f)

with open("pybo/model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pybo/model/cluster_names.pkl", "rb") as f:
    cluster_names = pickle.load(f)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():

    n_en = float(request.form["N_EN"])
    n_fat = float(request.form["N_FAT"])
    n_cho = float(request.form["N_CHO"])
    n_prot = float(request.form["N_PROT"])
    n_sugar = float(request.form["N_SUGAR"])

    fat_ratio = n_fat / n_en
    carb_ratio = n_cho / n_en
    protein_ratio = n_prot / n_en
    sugar_ratio = n_sugar / n_en

    x = np.array([[fat_ratio, carb_ratio, protein_ratio, sugar_ratio]])
    x_scaled = scaler.transform(x)

    cluster = model.predict(x_scaled)[0]
    result = cluster_names[cluster]

    return render_template("result.html", result=result)