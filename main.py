import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt

# === Настройка ===
CSV_PATH = "usage.csv"  # укажи путь к своему CSV

# === Шаг 1: Загрузка данных ===
df = pd.read_csv(CSV_PATH)

# === Шаг 2: Расчёт длительности и времени суток ===
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
    else:
        return "ночь"

df['start_dt'] = df['start_time'].apply(parse_time)
df['end_dt'] = df['end_time'].apply(parse_time)
df['duration_min'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60
df['period'] = df['start_dt'].apply(get_time_period)

# === Шаг 3: Кодировка признаков ===
le_day = LabelEncoder()
le_period = LabelEncoder()

df['day_code'] = le_day.fit_transform(df['day_of_week'])
df['period_code'] = le_period.fit_transform(df['period'])

# === Шаг 4: Обучение модели ===
features = df[['table_id', 'day_code', 'period_code']]
target = df['duration_min']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, target)

# === Шаг 5: Предсказание ===
def predict(table_id, day_of_week, time_period):
    day_code = le_day.transform([day_of_week])[0]
    period_code = le_period.transform([time_period])[0]
    X_input = [[table_id, day_code, period_code]]
    prediction = model.predict(X_input)[0]
    print(f"🔮 Прогноз: стол {table_id} в {day_of_week} ({time_period}) будет занят примерно {prediction:.1f} минут.")
    return prediction

# === Пример предсказаний ===
predict(3, "Пн", "вечер")
predict(1, "Сб", "день")
