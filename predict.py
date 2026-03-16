import joblib
import pandas as pd

try:
    model = joblib.load('model.v5')
except:
    print("Ошибка: Сначала запустите train.py")
    exit()

print("ТЕСТОВАЯ СИСТЕМА ПРЕДСКАЗАНИЯ CO2")

try:
    engine = float(input("Введите объем двигателя (L): "))
    cylinders = int(input("Введите кол-во цилиндров: "))
    fuel_l = float(input("Введите расход топлива (L/100 km): "))
    fuel_mpg = float(input("Введите расход топлива (mpg): "))

    input_data = pd.DataFrame(
        [[engine, cylinders, fuel_l, fuel_mpg]],
        columns=['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']
    )

    prediction = model.predict(input_data)
    print(f"\nРЕЗУЛЬТАТ ПРОВЕРКИ")
    print(f"Предсказанные выбросы: {prediction[0]:.2f} g/km")

except Exception as e:
    print(f"Ошибка ввода: {e}")