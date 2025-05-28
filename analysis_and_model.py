import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from ucimlrepo import fetch_ucirepo

def main():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    data = None
    option = st.radio("Источник данных:", 
                     ["Загрузить файл", "Демо-данные (требует интернет)", "Локальный файл (data/)"])
    
    if option == "Загрузить файл":
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
    
    elif option == "Демо-данные (требует интернет)":
        try:
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            st.success("Демо-данные загружены!")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    
    elif option == "Локальный файл (data/)":
        if os.path.exists("data/predictive_maintenance.csv"):
            data = pd.read_csv("data/predictive_maintenance.csv")
            st.success("Локальные данные загружены!")
        else:
            st.warning("Файл не найден: data/predictive_maintenance.csv")
    
    if data is None:
        st.warning("Загрузите данные для продолжения")
        return

    # Предобработка
    to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    cols_to_drop = [col for col in to_drop if col in data.columns]
    data = data.drop(columns=cols_to_drop)
    
    le = LabelEncoder()
    if 'Type' in data.columns:
        data['Type'] = le.fit_transform(data['Type'])
    
    # Разделение данных
    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader("Результаты оценки")
    st.write(f"Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Предсказание')
    ax.set_ylabel('Факт')
    st.pyplot(fig)
    
    # Прогнозирование
    st.subheader("Прогнозирование на новых данных")
    with st.form("prediction_form"):
        inputs = {
            'Type': st.selectbox("Тип оборудования", options=['L','M','H']),
            'Air temperature [K]': st.number_input("Температура воздуха (K)", value=300.0),
            'Process temperature [K]': st.number_input("Температура процесса (K)", value=310.0),
            'Rotational speed [rpm]': st.number_input("Скорость вращения (rpm)", value=1500),
            'Torque [Nm]': st.number_input("Крутящий момент (Nm)", value=40.0),
            'Tool wear [min]': st.number_input("Износ инструмента (min)", value=0)
        }
        
        if st.form_submit_button("Предсказать"):
            input_df = pd.DataFrame([inputs])
            input_df['Type'] = le.transform(input_df['Type'])
            
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)[0][1]
            
            if prediction[0] == 1:
                st.error(f"Прогноз: Отказ оборудования (вероятность: {proba:.2%})")
            else:
                st.success(f"Прогноз: Оборудование исправно (вероятность: {1-proba:.2%})")

if __name__ == "__main__":
    main()