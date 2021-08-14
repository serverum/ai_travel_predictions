# coding=utf8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

trips_data = pd.read_excel("trips_data.xls")
print(trips_data.head(5))  # выводит первые n строк из таблицы, head(n)
print(trips_data.describe())  # describe() - отображает суммарную инфу по текущей таблице(кол-во строк, среднее и т.п.)
print(trips_data.shape)  # смотрим форму таблицы, т.е. (сколько строк, столбцев) - (1000 строк, 7 столбцев)
print(trips_data['salary'])  # обращаемся к конкретному значению таблицы, а именно, к всей колонке зарплат
print(trips_data['salary'].describe())  # так же можем смотреть по каждой колонке через метод describe()
trips_data["salary"].plot()  # вывод графика с ценами, нужен матплотлиб
trips_data["salary"].hist()  # вывод гистограммы с зарплатами, нужно импортировать матплотлиб
print(
    trips_data['city'].value_counts())  # считает дискретные данные, т.е. отображает кол-во городов в таблице по кол-ву
print(trips_data.columns)  # выводит кол-во колонов в таблице
trips_data['city'].value_counts().plot(kind="bar")  # отображает данные в диаграмме в виде планок по найбольшему кол-ву
trips_data.groupby("city")['salary'].mean()  # Группировка по городам и зарплате, берем среднюю цифру

# Начало работы с машинным обучением sklearn randomforestclassifier

df = pd.get_dummies(trips_data, columns=['city', 'vacation_preference', 'transport_preference'])  # изучить подробно

X = df.drop("target", axis=1)  # убирает колонку таргет с таблицы, axis=1 это формат столбца в форме

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()  # можно указать настройки модели
model.fit(X_train, y_train)

# example = {col: [0] for col in X.columns}
example = {'salary': [80000], 'age': [31], 'family_members': [0], 'city_Екатеринбург': [0], 'city_Киев': [0],
           'city_Краснодар': [0], 'city_Минск': [0], 'city_Москва': [0], 'city_Новосибирск': [0], 'city_Омск': [0],
           'city_Петербург': [0], 'city_Томск': [0], 'city_Хабаровск': [0], 'city_Ярославль': [0],
           'vacation_preference_Архитектура': [1], 'vacation_preference_Ночные клубы': [0],
           'vacation_preference_Пляжный отдых': [0], 'vacation_preference_Шоппинг': [0],
           'transport_preference_Автомобиль': [0], 'transport_preference_Космический корабль': [0],
           'transport_preference_Морской транспорт': [0], 'transport_preference_Поезд': [0],
           'transport_preference_Самолет': [1]}

example_df = pd.DataFrame(example)
# print(example)
print(model.predict(example_df))  # предсказывает на основании данных, конечный вариант
print(model.predict_proba(example_df))  # выводит вероятность с которой выбирает, в нашем случае города
print(model.classes_)  # выводит названия всех классов для выборки, т.е. конечный результат выбора

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

print(len(X_test))
