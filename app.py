import os
import random
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import warnings

# --- १. लायब्ररी इम्पोर्ट सुधारला (Error: module 'google.genai' has no attribute 'configure' फिक्स झाला) ---
import google.generativeai as genai 

sensor_data = {"temperature": 0, "humidity": 0, "rainfall": 0}
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    crop = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

with app.app_context():
    db.create_all()

# --- २. ML मॉडेल लोड करणे ---
try:
    model = joblib.load("model.pkl")
    ms = joblib.load("minmaxscaler.pkl")
    sc = joblib.load("standscaler.pkl")
except Exception as e:
    print(f"Error loading models: {e}")

# --- ३. लॉगिन/साइनअप रूट्स ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user: return "Username already exists! Please login."
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            error = "Invalid username or password!"
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    user_history = History.query.filter_by(user_id=current_user.id).all()
    return render_template('index.html', history=user_history[::-1])

# --- ४. IoT सेंसर API ---
latest_sensor_data = {"temperature": 0.0, "humidity": 0.0, "rainfall": 0.0}

@app.route('/api/sensor-data', methods=['GET', 'POST'])
def handle_sensor_data():
    global latest_sensor_data
    if request.method == 'POST':
        data = request.get_json()
        latest_sensor_data.update(data)
        return jsonify({"status": "success"}), 200
    return jsonify(latest_sensor_data)

# --- ५. क्रॉप प्रेडिक्शन ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_features = np.array([[
            float(data['nitrogen']), float(data['phosphorus']), float(data['potassium']),
            float(data['temperature']), float(data['humidity']), float(data['ph']), float(data['rainfall'])
        ]])
        scaled_features = ms.transform(raw_features)
        prediction = model.predict(scaled_features)[0]
        
        crop_dict = {0: "Apple", 1: "Banana", 2: "Blackgram", 3: "Chickpea", 4: "Coconut", 5: "Coffee", 6: "Cotton", 7: "Grapes", 8: "Jute", 9: "Kidneybeans", 10: "Lentil", 11: "Maize", 12: "Mango", 13: "Mothbeans", 14: "Mungbean", 15: "Muskmelon", 16: "Orange", 17: "Papaya", 18: "Pigeonpeas", 19: "Pomegranate", 20: "Rice", 21: "Watermelon", 22: "Wheat", 23: "Sugarcane"}
        
        final_crop = crop_dict.get(int(prediction), "Unknown")
        if current_user.is_authenticated:
            db.session.add(History(user_id=current_user.id, crop=final_crop))
            db.session.commit()
        return jsonify({"crop": final_crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ६. रिअल AI ॲग्री-बॉट (Gemini Integration - FIXED) ---
# Render वरील पर्यावरणातून की वाचली जाते
API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
chatbot_model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_msg = request.json.get('message')
        prompt = f"You are 'Agri-Bot'. Answer STRICTLY in MARATHI only. Question: {user_msg}"
        # नवीन अचूक सिंटॅक्स
        response = chatbot_model.generate_content(prompt)
        return jsonify({"reply": response.text.replace("**", "")})
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({"reply": "क्षमस्व, माझे AI मेंदू सध्या विश्रांती घेत आहे!"}), 500

# --- ७. एक्स्ट्रा सेंसर रूट्स (Indentation Fixed) ---
@app.route('/update-sensor', methods=['POST'])
def update_sensor():
    global sensor_data
    sensor_data = request.get_json()
    return {"message": "Data received successfully"}, 200

@app.route('/get-data', methods=['GET'])
def get_data():
    return jsonify(sensor_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
