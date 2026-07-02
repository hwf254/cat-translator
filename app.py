"""
貓語翻譯 - Flask 後端 API

端點:
  POST /analyze   接收音檔(form-data, 欄位名 audio),回傳判斷結果 JSON
  GET  /           回傳前端錄音頁面

部署到 Render 時記得:
  1. requirements.txt 一起上傳
  2. Start command: gunicorn app:app
  3. ffmpeg 要能用(Render 預設環境通常有,如果沒有要加 buildpack)
"""

import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from cat_meow_classifier import analyze

app = Flask(__name__, static_folder="static")
CORS(app)  # 手機瀏覽器跨網域打 API 需要

ALLOWED_EXT = {"wav", "webm", "mp3", "m4a", "ogg"}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "沒有收到音檔,欄位名要是 audio"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "檔名是空的"}), 400

    # 存成暫存檔給 librosa 讀
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = analyze(tmp_path)
        # features 裡的 numpy float 轉成 python float,不然 jsonify 會出錯
        result["features"] = {k: float(v) for k, v in result["features"].items()}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"分析失敗: {str(e)}"}), 500
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
