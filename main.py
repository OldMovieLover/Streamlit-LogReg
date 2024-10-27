import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


st.write("""# Исследование логистической регресии""")

file = st.file_uploader("Загрузите CSV-файл", type="csv")
if file is not None:
    train = pd.read_csv(file)
    st.write(train.head(5))


    class LogReg:
        def __init__(self, learning_rate=0.1, epochs=10):
            self.learning_rate = learning_rate
            self.epochs = epochs

        def fit(self, X, y):
            X= np.array(X)
            y= np.array(y)
            self.coef_ = np.random.uniform(-1, 1, size=X.shape[1])
            self.intercept_ = np.random.uniform(-1, 1) 

            for epoch in range(self.epochs):
                y_pred = self.predict(X)
                error = (y - y_pred)
                grad_w_0 = - error
                grad_w = - X * error.reshape(-1, 1)

                self.coef_ -= self.learning_rate * grad_w.mean(axis=0)
                self.intercept_ -= self.learning_rate * grad_w_0.mean()

        def sigmoid(self, z):
            return 1 / (1 + (np.exp(-z)))

        def predict(self, X):
            y_pred = self.sigmoid(X @ self.coef_ + self.intercept_)
            return y_pred
        
        def score(self, X, y):
            y_pred = self.predict(X) >= 0.5
            return accuracy_score(y, y_pred)
        
        def get_weights(self, feature_names):
            weights = {name: weight for name, weight in zip(feature_names, self.coef_)}
            weights['intercept'] = self.intercept_ 
            return weights
        
    features = train.columns[:-1]  
    target_column = train.columns[-1]  

    learning_rate = st.text_input("Введите скорость обучения")
    if learning_rate:
        try:
            learning_rate = float(learning_rate)
        except ValueError:
            st.error("Пожалуйста, введите корректное значение для скорости обучения.")
    else:
       learning_rate = None

    if learning_rate is not None:
        epochs = st.text_input("Введите количество эпох обучения")
        if epochs:
            try:
                epochs = int(epochs)
            except ValueError:
                st.error("Пожалуйста, введите корректное значение для эпох.")
        else:
            epochs = None

        if epochs is not None:
    
            ss = StandardScaler()
            y_train_column = train.columns[-1]
            X_train, y_train = train.drop(y_train_column, axis=1), train[y_train_column]
            X_train = ss.fit_transform(X_train)

            st.write("## Результаты с применением логистической регрессии и сравнение результата с логистической регрессией в sklearn")
            if st.button("Обучить модель"):
                X_train = train[features].values
                y_train = train[target_column].values

                model = LogReg(learning_rate, epochs)
                model.fit(X_train, y_train)
                
                weights = model.get_weights(features)
                y_pred = model.predict(X_train) >= 0.5
                custom_accuracy = accuracy_score(y_train, y_pred)

                sklearn_model = LogisticRegression()
                sklearn_model.fit(X_train, y_train)
                sklearn_weights = {name: weight for name, weight in zip(features, sklearn_model.coef_[0])}
                sklearn_weights['intercept'] = sklearn_model.intercept_[0]
                sklearn_y_pred = sklearn_model.predict(X_train)
                sklearn_accuracy = accuracy_score(y_train, sklearn_y_pred)

                st.write("Коэффициенты (веса) вашей модели:")
                st.json(weights)

                st.write("Коэффициенты (веса) модели scikit-learn:")
                st.json(sklearn_weights)

                st.write(f"Точность вашей модели: {custom_accuracy:.4f}")
                st.write(f"Точность модели scikit-learn: {sklearn_accuracy:.4f}")


            st.write("## Scatter Plot Visualization")
            st.write("Выберите две фичи для построения графика")

            feature_x = st.selectbox("Выберите первую фичу", features)
            feature_y = st.selectbox("Выберите вторую фичу", features)

            if st.button("Построить график"):
                
                x = train[feature_x]
                y = train[feature_y]
                target = train[target_column]

                
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(x, y, c=target, cmap='bwr', alpha=0.6)
                plt.xlabel(feature_x)
                plt.ylabel(feature_y)
                plt.title("Scatter Plot of {} vs {}".format(feature_x, feature_y))
                plt.colorbar(scatter, label=target_column)
                plt.grid()
                
                st.pyplot(plt)
        
        else:
            st.stop()
        
    else:
        st.stop()

else:
    st.stop()