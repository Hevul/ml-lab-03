import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import graphviz

# Загрузка данных
df = pd.read_csv("citrus.csv")

# Выделение признаков и целевой переменной
X = df.drop(columns=['name'])
y = df['name']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("Матрица ошибок:")
    print(conf_matrix)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    return accuracy

# Дерево решений
print("Дерево решений:")
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train, y_train)
accuracy_dt = evaluate_model(base_model, X_test, y_test)

# Подбор гиперпараметров для дерева решений
param_grid_dt = {"max_depth": np.arange(1, 11), "max_features": [0.5, 0.7, 1]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)
print("Лучшие параметры для дерева решений:", grid_search_dt.best_params_)
best_model_dt = grid_search_dt.best_estimator_
accuracy_dt_best = evaluate_model(best_model_dt, X_test, y_test)
print(f'Улучшение точности (Decision Tree): {accuracy_dt_best - accuracy_dt:.2f}')

# Метод k-ближайших соседей (KNN)
print("K-ближайших соседей (KNN):")
base_model_knn = KNeighborsClassifier()
base_model_knn.fit(X_train, y_train)
accuracy_knn = evaluate_model(base_model_knn, X_test, y_test)

# Подбор гиперпараметров для KNN
param_grid_knn = {"n_neighbors": np.arange(1, 20)}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
print("Лучшие параметры для KNN:", grid_search_knn.best_params_)
best_model_knn = grid_search_knn.best_estimator_
accuracy_knn_best = evaluate_model(best_model_knn, X_test, y_test)
print(f'Улучшение точности (KNN): {accuracy_knn_best - accuracy_knn:.2f}')

# Случайный лес (Random Forest)
print("Случайный лес (Random Forest):")
base_model_rf = RandomForestClassifier(random_state=42)
base_model_rf.fit(X_train, y_train)
accuracy_rf = evaluate_model(base_model_rf, X_test, y_test)

# Подбор гиперпараметров для случайного леса
param_grid_rf = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
print("Лучшие параметры для Random Forest:", grid_search_rf.best_params_)
best_model_rf = grid_search_rf.best_estimator_
accuracy_rf_best = evaluate_model(best_model_rf, X_test, y_test)
print(f'Улучшение точности (Random Forest): {accuracy_rf_best - accuracy_rf:.2f}')

# Выводы о лучшей модели
print("\nСравнение моделей:")
print(f"Decision Tree: {accuracy_dt_best:.2f}")
print(f"KNN: {accuracy_knn_best:.2f}")
print(f"Random Forest: {accuracy_rf_best:.2f}")

best_model_name = max(
    {"Decision Tree": accuracy_dt_best, "KNN": accuracy_knn_best, "Random Forest": accuracy_rf_best},
    key=lambda k: {"Decision Tree": accuracy_dt_best, "KNN": accuracy_knn_best, "Random Forest": accuracy_rf_best}[k]
)
print(f"Лучшая модель: {best_model_name}")

# Визуализация дерева
model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train, y_train)

dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X.columns,  # Имена признаков
    class_names=y.unique(),    # Имена классов
    filled=True,              # Заливка цветом
    rounded=True,             # Закругленные углы
    special_characters=True   # Специальные символы
)

# Визуализация дерева
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # Сохранение в файл
graph.view("decision_tree")  # Открытие визуализации