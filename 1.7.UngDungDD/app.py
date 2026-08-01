from flask import Flask
from api.api import api   # 👈 import từ thư mục api

app = Flask(__name__)

# đăng ký API (blueprint)
app.register_blueprint(api)

@app.route("/")
def home():
    return {
        "message": "Lung Cancer Prediction API is running"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)