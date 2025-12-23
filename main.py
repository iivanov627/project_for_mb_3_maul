import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("students.csv")

print("Первые 5 строк датасета:")
print(df.head())


X = df[['ParentalEducation', 'StudyTimeWeekly', 'Absences']]
y = df['GPA']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nОценка модели:")
print(f"MSE (среднеквадратичная ошибка): {mse:.4f}")
print(f"R² (коэффициент детерминации): {r2:.4f}")


coefficients = pd.DataFrame({
    'Фактор': X.columns,
    'Коэффициент': model.coef_
})

print("\nКлючевые факторы успеха:")
print(coefficients)

intercept = model.intercept_

print("\nУравнение множественной регрессии:")
print(
    f"GPA = {intercept:.3f} "
    f"+ ({model.coef_[0]:.3f}) * ParentalEducation "
    f"+ ({model.coef_[1]:.3f}) * StudyTimeWeekly "
    f"+ ({model.coef_[2]:.3f}) * Absences"
)


new_student = pd.DataFrame({
    'ParentalEducation': [4],
    'StudyTimeWeekly': [30],
    'Absences': [5]
})

predicted_gpa = model.predict(new_student)

print("\nПример предсказания:")
print(f"Предсказанный GPA студента: {predicted_gpa[0]:.2f}")
