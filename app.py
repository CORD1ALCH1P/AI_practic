from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pickle
import os

app = Flask(__name__)

# Загрузка и подготовка данных
def load_and_prepare_data():
    CSV_PATH = "exm.csv"
    df = pd.read_csv(CSV_PATH)
    
    def parse_time(t):
        return datetime.strptime(t, "%H:%M")

def get_time_period(t):
    h = t.hour
    if 6 <= h < 12:
        return "утро"
    elif 12 <= h < 18:
        return "день"
    elif 18 <= h < 24:
        return "вечер"
    else:  # 0 <= h < 6
        return "ночь"

    df['start_dt'] = df['start_time'].apply(parse_time)
    df['end_dt'] = df['end_time'].apply(parse_time)
    df['duration_min'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60
    df['period'] = df['start_dt'].apply(get_time_period)

    le_day = LabelEncoder()
    le_period = LabelEncoder()

    df['day_code'] = le_day.fit_transform(df['day_of_week'])
    df['period_code'] = le_period.fit_transform(df['period'])

    return df, le_day, le_period

# Обучение модели
def train_model(df):
    features = df[['table_id', 'day_code', 'period_code']]
    target = df['duration_min']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    
    return model

# Проверяем, есть ли сохраненная модель
if not os.path.exists('model.pkl') or not os.path.exists('encoders.pkl'):
    df, le_day, le_period = load_and_prepare_data()
    model = train_model(df)
    
    # Сохраняем модель и кодировщики
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('encoders.pkl', 'wb') as f:
        pickle.dump((le_day, le_period), f)
else:
    # Загружаем модель и кодировщики
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('encoders.pkl', 'rb') as f:
        le_day, le_period = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    table_id = int(data['table_id'])
    day_of_week = data['day_of_week']
    time_period = data['time_period']
    
    try:
        day_code = le_day.transform([day_of_week])[0]
        period_code = le_period.transform([time_period])[0]
        X_input = [[table_id, day_code, period_code]]
        prediction = model.predict(X_input)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 1),
            'table_id': table_id,
            'day_of_week': day_of_week,
            'time_period': time_period
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)