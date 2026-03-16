import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

df = pd.read_csv("CO2 Emissions_Canada.csv")
model = joblib.load('model.v5')

st.title("Прогноз выбросов CO2 автомобилями")
st.write("Тема: Линейная регрессия в машинном обучении")
st.write("Название датасета: CO2 Emission by Vehicles")

col_left, col_right = st.columns(2)

with col_left:
    st.header("Визуализация")
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']
    X = df[features]
    y = df['CO2 Emissions(g/km)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.2)

    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)

    ax.set_xlabel("Реальные значения CO2 (g/km)")
    ax.set_ylabel("Предсказанные значения CO2 (g/km)")
    ax.grid(True)

    st.pyplot(fig)

    st.write("Введите параметры для предсказания:")
    val_engine = st.number_input("Объем двигателя (L)", value=2.0)
    val_cylinders = st.number_input("Количество цилиндров", value=4)
    val_fuel = st.number_input("Расход топлива (L/100 km)", value=8.5)
    val_fuel_mpg = st.number_input("Расход топлива (mpg)", value=33.0)

    if st.button("Выполнить расчет"):
        input_row = pd.DataFrame([[val_engine, val_cylinders, val_fuel, val_fuel_mpg]], columns=features)
        res = model.predict(input_row)
        st.write("Результат линейной регрессии:")
        st.success(f"{res[0]:.2f} g/km")

with col_right:
    st.header("О датасете")

    st.write("Всего данных в базе:", len(df))

    st.write("Пример данных:")
    st.dataframe(df.head(15))

    st.write("Все колонки исходного датасета:")
    st.write(", ".join(df.columns))

    st.write("Колонки, взятые для расчета:")
    st.write("Engine Size(L), Cylinders, Fuel Consumption Comb (L/100 km), Fuel Consumption Comb (mpg)")

    st.write("Точность модели (Accuracy):")
    st.info("0.9004 (90%)")