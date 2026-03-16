import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("CO2 Emissions_Canada.csv")

features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']
X = df[features]
y = df['CO2 Emissions(g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"РЕЗУЛЬТАТ")
print(f"Accuracy: {accuracy:.4f}")

joblib.dump(model, 'model.v5')
print("Модель сохранена как model.v5")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.3, label='Реальные данные')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=3, label='Линия регрессии')
plt.title(f"Linear Regression (Accuracy: {accuracy:.4f})")
plt.xlabel("Реальные выбросы CO2")
plt.ylabel("Предсказанные выбросы CO2")
plt.legend()
plt.grid(True)
plt.show()