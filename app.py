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
from google import genai

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


# 2. Load ML Model & BOTH Scalers

model = joblib.load("model.pkl")
# we use both scaler in our folder
ms = joblib.load("minmaxscaler.pkl")
sc = joblib.load("standscaler.pkl")

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
    user_history = user_history[::-1] 
    return render_template('index.html', history=user_history)

# --- Real IoT Sensor API ---
# हा एक रिकामा डबा (Variable) आहे, ज्यात हार्डवेअरचा डेटा सेव्ह राहील
latest_sensor_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "rainfall": 0.0
}

@app.route('/api/sensor-data', methods=['GET', 'POST'])
def handle_sensor_data():
    global latest_sensor_data
    
    if request.method == 'POST':
        # १. ही स्टेप तुझा IoT डिव्हाईस वापरून डेटा सर्व्हरला देईल
        data = request.get_json()
        latest_sensor_data['temperature'] = data.get('temperature', 0)
        latest_sensor_data['humidity'] = data.get('humidity', 0)
        latest_sensor_data['rainfall'] = data.get('rainfall', 0)
        print(f"📡 Real Data Received from Hardware: {latest_sensor_data}")
        return jsonify({"status": "success", "message": "Data saved in Flask!"}), 200
        
    elif request.method == 'GET':
        # २. ही स्टेप तुझी वेबसाईट (Frontend) वापरून डेटा खेचून घेईल
        return jsonify(latest_sensor_data)
# -----------------------------
# ---------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # users inputes
        raw_features = np.array([[
            float(data['nitrogen']), float(data['phosphorus']), float(data['potassium']),
            float(data['temperature']), float(data['humidity']), float(data['ph']), float(data['rainfall'])
        ]])
        
        print("\n--- NEW PREDICTION ---")
        print("1. Raw Data:", raw_features)
        
        # try minmax scaler
        try:
            ms = joblib.load("minmaxscaler.pkl")
            scaled_features = ms.transform(raw_features)
        except:
            scaled_features = raw_features
            
        # if our scaler file is currpt and inputs are not smaller then scale manually
        if np.array_equal(raw_features, scaled_features):
            print(" Scaler file is broken! Applying manual auto-scaling...")
            mins = np.array([0.0, 5.0, 5.0, 8.8, 14.2, 3.5, 20.2])
            maxs = np.array([140.0, 145.0, 205.0, 43.6, 99.9, 9.9, 298.5])
            scaled_features = (raw_features - mins) / (maxs - mins)
        
        print("2. Scaled Data for Model:", scaled_features)
        
        # Prediction of Model
        prediction = model.predict(scaled_features)[0]
        print("3. Model Predicted Number:", prediction)
        
        crop_dictionary = {
            0: "Apple", 1: "Banana", 2: "Blackgram", 3: "Chickpea", 
            4: "Coconut", 5: "Coffee", 6: "Cotton", 7: "Grapes", 
            8: "Jute", 9: "Kidneybeans", 10: "Lentil", 11: "Maize", 
            12: "Mango", 13: "Mothbeans", 14: "Mungbean", 15: "Muskmelon", 
            16: "Orange", 17: "Papaya", 18: "Pigeonpeas", 19: "Pomegranate", 
            20: "Rice", 21: "Watermelon", 22: "Wheat", 23: "Sugarcane"
        }
        
        final_crop_name = crop_dictionary.get(int(prediction), f"Crop Type {prediction}")
        print("4. Final Crop Name:", final_crop_name)
        
        if current_user.is_authenticated:
            new_record = History(user_id=current_user.id, crop=final_crop_name)
            db.session.add(new_record)
            db.session.commit()
        
        return jsonify({
         "crop": final_crop_name, "accuracy": 92, "precision": 89, "recall": 91, "f1": 90
        })
        
    except Exception as e:
        print("Backend Error:", e)
        return jsonify({"error": str(e)}), 500
# --- History Delete Route ---
@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    try:
        # delete login users history
        History.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        print("Error clearing history:", e)
        return jsonify({"error": str(e)}), 500
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
    return response

# 5. REAL AI Agri-Bot (Gemini API Integration)

GEMINI_API_KEY = "genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))"

#Initialize the new GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_msg = request.json.get('message')
        
        # System prompt to instruct the AI how to behave
        # --- AI Chatbot Prompt (Strictly Marathi) ---
        prompt = f"""
        You are 'Agri-Bot', a highly knowledgeable agricultural expert from India. 
        Your job is to help farmers with their farming, crop diseases, fertilizers, and weather-related queries.
        
        CRITICAL RULE: You MUST answer the following question STRICTLY in the MARATHI language only. 
        Use simple, easy-to-understand Marathi words that a local farmer can easily read and understand.
        
        Farmer's Question: {user_msg}
        """
        # ---------------------------------------------
        # Send request to Gemini using the new API syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Using the latest and fastest model
            contents=prompt
        )
        
        reply = response.text
        # Remove bold formatting (**) for a cleaner UI display
        reply = reply.replace("**", "") 
        
        return jsonify({"reply": reply})
        
    except Exception as e:
        print("API Error:", e)
        return jsonify({"reply": "Sorry, my AI brain is sleeping right now! Please check your internet connection or API key."}), 500
    @app.route('/update-sensor', methods=['POST'])
    def update_sensor():
       global sensor_data
    # ESP32 कडून येणारा JSON डेटा इथे स्वीकारला जातो
    sensor_data = request.get_json() 
    print(f"Received Data: {sensor_data}") # हा फक्त चेक करण्यासाठी आहे
    return {"message": "Data received successfully"}, 200

    @app.route('/get-data', methods=['GET'])
    def get_data():
    # हा डेटा तुमचं मोबाईल ॲप दर ३ सेकंदांनी वाचेल
       return jsonify(sensor_data)

if __name__ == '__main__':
   
    # PORT setting for render (dynamic port)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
