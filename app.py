import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify, send_from_directory
from chat_server import load_or_generate_vectors, get_answer

app = Flask(__name__)
vectors = load_or_generate_vectors()

# OSRM API to calculate road distance
def get_road_distance(lat1, lon1, lat2, lon2):
    url = f'http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false'
    try:
        response = requests.get(url)
        data = response.json()
        distance = data['routes'][0]['legs'][0]['distance'] / 1000  # Convert to kilometers
        return distance
    except Exception as e:
        print(f"Error getting road distance: {e}")
        return None

# Read CSV for hospital data
df_hospitals = pd.read_csv('hospitals.csv')

# Make sure to have the correct columns and apply preprocessing for hospital data
le_weather = LabelEncoder()
le_traffic = LabelEncoder()
df_hospitals['weather_encoded'] = le_weather.fit_transform(df_hospitals['weather'])
df_hospitals['traffic_encoded'] = le_traffic.fit_transform(df_hospitals['traffic'])

# Set traffic factors for calculating time to reach
traffic_factor = {'low': 1, 'medium': 1.25, 'high': 1.5}

# Define calculate_time_to_reach function
def calculate_time_to_reach(row, user_lat, user_lon):
    distance = get_road_distance(user_lat, user_lon, row['hospital_lat'], row['hospital_lon'])
    if distance is None:
        return None
    traffic = row['traffic']
    average_speed = row['average_speed']
    eta = (distance / average_speed) * 60 * traffic_factor.get(traffic, 1)
    return eta

# Apply the function to calculate the time to reach each hospital
df_hospitals['time_to_reach'] = df_hospitals.apply(lambda row: calculate_time_to_reach(row, user_lat=12.9716, user_lon=80.2750), axis=1)

# Create the 'on_time' column based on the threshold (30 minutes here)
df_hospitals['on_time'] = (df_hospitals['time_to_reach'] <= 30).astype(int)

# Prepare features (X) and target (y) for logistic regression
X = df_hospitals[['hospital_lat', 'hospital_lon', 'weather_encoded', 'traffic_encoded', 'average_speed']]
y = df_hospitals['on_time']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Load the disease dataset
df_diseases = pd.read_excel('project.xlsx', sheet_name='No_of_Diseases')

# Define the mappings for each parameter (ECG, Temp, Pulse, BP, SpO2)
ECG_MAPPING = { 
    'low': -0.1, 
    'normal': 0, 
    'high': 0.1
}

TEMP_MAPPING = {
    32: -0.5, 33: -0.4, 34: -0.3, 35: -0.2, 36: -0.1, 36.12: 0, 
    37: 0.1, 38: 0.2, 39: 0.3, 40: 0.4, 41: 0.5
}

PULSE_MAPPING = {
    29: -0.4, 37: -0.3, 46: -0.2, 54: -0.1, 80: 0, 110: 0.1, 
    150: 0.2, 215: 0.3, 250: 0.4
}

SPO2_MAPPING = {
    64: -0.5, 72: -0.4, 86: -0.3, 93: -0.2, 97: 0, 101: 0.1
}

# Define a mapping for BP based on systolic and diastolic as a tuple (systolic, diastolic)
BP_PAIR_MAPPING = {
    (85, 55): -0.6, (90, 60): -0.5, (95, 65): -0.4, (100, 70): -0.3, 
    (110, 75): -0.2, (115, 80): -0.1, (120, 80): 0, (135, 85): 0.1, 
    (140, 90): 0.2, (150, 95): 0.3, (165, 100): 0.4, (175, 105): 0.5, (185, 110): 0.6
}

# Helper function to get the nearest value
def get_nearest_value(value, param_mapping, param_name):
    if param_name == 'ECG':  # Handling ECG based on ranges
        if value < 120:
            return ECG_MAPPING['low']
        elif 120 <= value <= 200:
            return ECG_MAPPING['normal']
        else:
            return ECG_MAPPING['high']
    elif param_name == 'BP':  # Handling BP as pair (systolic, diastolic)
        nearest_value = min(param_mapping, key=lambda k: abs(k[0] - value[0]) + abs(k[1] - value[1]))
        return param_mapping[nearest_value]
    else:  # For other parameters
        nearest_value = min(param_mapping, key=lambda k: abs(k - value))
        return param_mapping[nearest_value]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hospital_app')
def hosp():
    return render_template('hosp.html')

@app.route('/predict_hospital', methods=['POST'])
def predict_hospital():
    try:
        user_lat = float(request.form['user_lat'])
        user_lon = float(request.form['user_lon'])
        heart_rate = int(request.form['heart_rate'])
        fatigue_level = request.form['fatigue_level']
        time_of_query = request.form['time_of_query']

        if not user_lat or not user_lon or not heart_rate or not fatigue_level or not time_of_query:
            return jsonify({'error': 'All fields are required!'})

        # Calculate the nearest hospital and prediction
        nearest_hospitals = []
        for index, row in df_hospitals.iterrows():
            distance = get_road_distance(user_lat, user_lon, row['hospital_lat'], row['hospital_lon'])
            if distance is None:
                continue  # Skip this hospital if we can't get a road distance
            weather = row['weather']
            traffic = row['traffic']
            average_speed = row['average_speed']
            weather_encoded = le_weather.transform([weather])[0]
            traffic_encoded = le_traffic.transform([traffic])[0]
            input_data = pd.DataFrame([[row['hospital_lat'], row['hospital_lon'], weather_encoded, traffic_encoded, average_speed]],
                                      columns=['hospital_lat', 'hospital_lon', 'weather_encoded', 'traffic_encoded', 'average_speed'])
            prob = model.predict_proba(input_data)[0][1]
            eta = calculate_time_to_reach(row, user_lat, user_lon)
            nearest_hospitals.append({
                'hospital_name': row['hospital_name'],
                'facilities': ' '.join(row['facilities']),
                'hospital_lat': row['hospital_lat'],
                'hospital_lon': row['hospital_lon'],
                'suburb': row['suburb'],
                'predicted_probability': prob,
                'weather': weather,
                'traffic': traffic,
                'average_speed': average_speed,
                'eta': eta,
                'distance': distance
            })

        best_hospital = min(nearest_hospitals, key=lambda x: x['eta'])

        result_data = {
            "hospital_info": f"The nearest hospital to your location is: {best_hospital['hospital_name']}\nCoordinates of your location: ({user_lat}, {user_lon})\nDriver's Heart Rate: {heart_rate} BPM\nDriver's Fatigue Level: {fatigue_level}\nTime of Query: {time_of_query}\n",
            "hospital_details": f"Hospital Locality: {best_hospital['suburb']}\nHospital Name: {best_hospital['hospital_name']}\nFacilities: {best_hospital['facilities']}\nWeather: {best_hospital['weather']}, Traffic: {best_hospital['traffic']}, Average Speed: {best_hospital['average_speed']} km/h\nDistance: {best_hospital['distance']:.2f} km\nETA: {best_hospital['eta']:.2f} minutes\n",
            "notification": f"Notification Sent\nPatient is en route to {best_hospital['hospital_name']}\nETA: {best_hospital['eta']:.2f} minutes"
        }

        return render_template('result.html', result_data=result_data)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        # Get user input
        ecg = float(request.form["ecg"])
        temp = float(request.form["temp"])
        pulse = int(request.form["pulse"])
        bp_systolic = int(request.form["bp_systolic"])
        bp_diastolic = int(request.form["bp_diastolic"])
        spo2 = int(request.form["spo2"])

        # Map the input values to their respective values in the table
        ecg_val = get_nearest_value(ecg, ECG_MAPPING, 'ECG')
        temp_val = get_nearest_value(temp, TEMP_MAPPING, 'TEMP')
        pulse_val = get_nearest_value(pulse, PULSE_MAPPING, 'PULSE')
        bp_val = get_nearest_value((bp_systolic, bp_diastolic), BP_PAIR_MAPPING, 'BP')
        spo2_val = get_nearest_value(spo2, SPO2_MAPPING, 'SPO2')

        # Check if all the parameters are in the set of -0.1, 0, 0.1
        normal_values = {-0.1, 0, 0.1}
        if all(val in normal_values for val in [ecg_val, temp_val, pulse_val, bp_val, spo2_val]):
            disease_name = "Normal"
            equation = f"{ecg_val} ECG + {temp_val} TEMP + {pulse_val} PULSE + {bp_val} BP + {spo2_val} SpO2"
        else:
            equation = f"{ecg_val} ECG + {temp_val} TEMP + {pulse_val} PULSE + {bp_val} BP + {spo2_val} SpO2"

            disease_name = "Normal"
            for index, row in df_diseases.iterrows():
                disease_params = [
                    row['ECG'], row['TEMP'], row['PULSE_RATE'], row['BLOOD_PRESSURE'], row['SpO2']
                ]
                if [ecg_val, temp_val, pulse_val, bp_val, spo2_val] == disease_params:
                    disease_name = row['Disease_Categories']
                    break

        # Calculate the 8-bit representation
        bits = [
            0 if ecg_val in [0, 0.1, -0.1] else 1,
            0 if temp_val in [0, 0.1, -0.1] else 1,
            0 if pulse_val in [0, 0.1, -0.1] else 1,
            0 if bp_val in [0, 0.1, -0.1] else 1,
            0 if bp_val in [0, 0.1, -0.1] else 1,
            0 if spo2_val in [0, 0.1, -0.1] else 1,
            0, 0  # The last 2 bits are always 0
        ]
        bit_representation = ''.join(map(str, bits))
        hex_representation = hex(int(bit_representation, 2))

        return render_template(
            "index2.html",
            equation=equation,
            disease_name=disease_name,
            bit_representation=bit_representation,
            hex_representation=hex_representation
        )

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/disease')
def app2():
    return render_template('index2.html')


@app.route("/chat")
def chat_ui():
    return send_from_directory("static/chat", "index.html")

@app.route("/chat/<path:filename>")
def chat_assets(filename):
    return send_from_directory("static/chat", filename)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = get_answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/dashboard")
def dashboard():
    return send_from_directory("static", "dashboard.html")

@app.route("/scenarios")
def scenarios():
    return send_from_directory("static", "scenarios.html")

if __name__ == "__main__":
    app.run(debug=True)
