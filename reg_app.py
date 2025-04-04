import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

@st.cache_data
def load_and_process_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['Месяц'] = pd.to_datetime(df['Месяц'], format='%d.%m.%Y')
    return df

uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xlsx"])
if uploaded_file:
    df = load_and_process_data(uploaded_file)
    df = df.sort_values('Месяц').reset_index(drop=True)

    df['Год'] = df['Месяц'].dt.year
    df['Месяц_номер'] = df['Месяц'].dt.month

    # Инициализация сессии, если данных еще нет
    if 'removed_data' not in st.session_state:
        st.session_state.removed_data = pd.DataFrame()  # Пустой DataFrame для удаленных данных
    if 'num_removed' not in st.session_state:
        st.session_state.num_removed = 0  # Количество удаленных строк
    if 'result_df' not in st.session_state:
        st.session_state.result_df = pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("В данных нет числовых столбцов для анализа!")
    else:
        selected_col = st.selectbox("**Выберите столбец для анализа**", numeric_cols, key="column_selector")

        # Создаем вкладки
        tab1, tab2, tab3 = st.tabs([
            "📊 Ящик с усами",
            "📈 Z-score метод",
            "📉 IQR метод"
        ])

        # Boxplot (Ящик с усами)
        with tab1:
            st.subheader("Анализ выбросов: Ящик с усами (Boxplot)")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(y=df[selected_col], ax=ax)
            ax.set_title(f"Распределение {selected_col}")
            st.pyplot(fig)

            # Вычисление выбросов
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            st.metric("Обнаружено выбросов", len(outliers))

            # Очистка всего датафрейма по IQR
            mask = ~((df[numeric_cols] < (df[numeric_cols].quantile(0.25) - 1.5 * (df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)))) |
                     (df[numeric_cols] > (df[numeric_cols].quantile(0.75) + 1.5 * (df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25))))).any(axis=1)
            num_removed = len(df) - mask.sum()
            removed_data = df[~mask]

            if st.button("Очистить весь датафрейм (Boxplot)", key="boxplot_clean"):
                st.session_state.removed_data = removed_data
                st.session_state.num_removed = num_removed
                st.session_state.result_df = df[mask]

        # Z-score метод
        with tab2:
            st.subheader("Анализ выбросов: Z-score метод")
            threshold = st.slider("Порог Z-score", 2.0, 5.0, 3.0, 0.1, key="z_slider")

            scaler = StandardScaler()
            z_scores = scaler.fit_transform(df[[selected_col]])
            df['z_score'] = np.abs(z_scores)

            outliers = df[df['z_score'] > threshold]

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df['z_score'], bins=30, kde=True, ax=ax)
            ax.axvline(x=threshold, color='r', linestyle='--')
            ax.set_title(f"Распределение Z-скоров ({selected_col})")
            st.pyplot(fig)

            st.metric("Обнаружено выбросов (Z > {threshold})", len(outliers))

            # Очистка всего датафрейма по Z-score
            z_scores_full = scaler.fit_transform(df[numeric_cols])
            mask = (np.abs(z_scores_full) <= threshold).all(axis=1)
            num_removed = len(df) - mask.sum()
            removed_data = df[~mask]

            if st.button("Очистить весь датафрейм (Z-score)", key="zscore_clean"):
                st.session_state.removed_data = removed_data
                st.session_state.num_removed = num_removed
                st.session_state.result_df = df[mask]


        # IQR метод
        with tab3:
            st.subheader("Анализ выбросов: IQR метод")
            iqr_multiplier = st.slider("Множитель IQR", 1.0, 3.0, 1.5, 0.1, key="iqr_slider")

            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

            fig, ax = plt.subplots(figsize=(10, 4))
            df[selected_col].plot(kind='line', ax=ax, alpha=0.5)
            ax.scatter(outliers.index, outliers[selected_col], color='red')
            ax.axhline(y=upper_bound, color='r', linestyle='--')
            ax.axhline(y=lower_bound, color='r', linestyle='--')
            ax.set_title(f"Выбросы в {selected_col} (IQR метод)")
            st.pyplot(fig)

            st.metric("Обнаружено выбросов", len(outliers))

            # Очистка всего датафрейма по IQR
            Q1_full = df[numeric_cols].quantile(0.25)
            Q3_full = df[numeric_cols].quantile(0.75)
            IQR_full = Q3_full - Q1_full
            lower_bound_full = Q1_full - iqr_multiplier * IQR_full
            upper_bound_full = Q3_full + iqr_multiplier * IQR_full

            mask = ~((df[numeric_cols] < lower_bound_full) | (df[numeric_cols] > upper_bound_full)).any(axis=1)
            num_removed = len(df) - mask.sum()
            removed_data = df[~mask]

            if st.button("Очистить весь датафрейм (IQR)", key="iqr_clean"):
                st.session_state.removed_data = removed_data
                st.session_state.num_removed = num_removed
                st.session_state.result_df = df[mask]

    if len(st.session_state.result_df) == 0:
        pass
    else:
        st.success(f"Удалено {st.session_state.num_removed} строк с выбросами!")
        st.dataframe(st.session_state.removed_data)
        df = st.session_state.result_df
        if 'z_score' in df.columns:
            df = df.drop(columns=['z_score'])
        result = pd.DataFrame({
            'Столбец': df.columns,
            'Процент пропусков (%)': df.isnull().mean() * 100
        })

        # Выводим с округлением
        st.write(result.round(2))

        option = st.selectbox(
            'Что хотите сделать с пропусками?',
            ['Заполнить медианой', 'Удалить строки с пропусками']
        )
        flag1 = False
        if option == 'Заполнить медианой':
            if st.button('Заполнить пропуски медианой'):
                flag1 = True
                df['AUSTRALIA-CHINA'] = df['AUSTRALIA-CHINA'].fillna(df['AUSTRALIA-CHINA'].median())
                df['AUSTRALIA-WORLD'] = df['AUSTRALIA-WORLD'].fillna(df['AUSTRALIA-WORLD'].median())
                df['PMI'] = df['PMI'].fillna(df['PMI'].median())
                df['BRAZIL-WORLD'] = df['BRAZIL-WORLD'].fillna(df['BRAZIL-WORLD'].median())
                df = df.drop(columns=['Boxit Guinea-China'])
                st.success("Пропуски успешно заполнены медианой!")


        elif option == 'Удалить строки с пропусками':
            if st.button('Удалить строки с пропусками'):
                flag1 = True
                df = df.dropna()
                st.success("Пропуски успешно удалены!")
        if flag1:
            correlations = df.corr(numeric_only=True)['C5TC'].sort_values(ascending=False)
            st.subheader("📈 Корреляция признаков с C5TC")
            st.dataframe(correlations.to_frame(name='Корреляция'), use_container_width=True)
            st.subheader("🔍 Тепловая карта корреляции")

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)



            columns_to_exclude = ['PMI', 'BRAZIL-WORLD', 'Месяц']
            df = df.drop(columns=columns_to_exclude)



            X = df.drop(columns=['C5TC'])  # Признаки
            y = df['C5TC']  # Целевая переменная

            # Разделим данные на тренировочную и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



            # Создадим и обучим модель линейной регрессии
            model = make_pipeline(
                StandardScaler(),
                LinearRegression()
            )
            model.fit(X_train, y_train)
            # model = LinearRegression()
            # model.fit(X_train, y_train)

            # Сделаем предсказания на тестовой выборке
            y_pred = model.predict(X_test)

            # Оценка модели
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Выводим результаты в Streamlit
            st.write(f'Mean Absolute Error (MAE): {mae}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')

            errors = y_test - y_pred

            # Показать график реальных значений против предсказанных
            st.subheader('Real vs Predicted C5TC (Line Plot)')
            plt.figure(figsize=(10, 6))
            plt.plot(y_test.reset_index(drop=True), label='Real Values', color='blue', alpha=0.7)
            plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
            plt.xlabel('Index')
            plt.ylabel('C5TC')
            plt.title('Real vs Predicted C5TC (Line Plot)')
            plt.legend()
            st.pyplot(plt)

            # Гистограмма ошибок
            st.subheader('Box Plot of Prediction Errors')
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=errors)
            plt.title('Box Plot of Prediction Errors')
            plt.xlabel('Error')
            st.pyplot(plt)

            # Сравнение распределений реальных и предсказанных значений
            st.subheader('Density Plot of Real vs Predicted C5TC')
            plt.figure(figsize=(8, 6))
            sns.kdeplot(y_test, label='Real Values', shade=True)
            sns.kdeplot(y_pred, label='Predicted Values', shade=True)
            plt.title('Density Plot of Real vs Predicted C5TC')
            plt.xlabel('C5TC')
            plt.ylabel('Density')
            plt.legend()
            st.pyplot(plt)

            print("Train Mean:", y_train.mean(), "Std:", y_train.std())
            print("Test Mean:", y_test.mean(), "Std:", y_test.std())


            df['Predicted C5TC'] = model.predict(X)

            # Сохранение DataFrame в Excel
            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_file.seek(0)

            # Кнопка для скачивания Excel
            st.download_button(
                label="Скачать Excel",
                data=excel_file,
                file_name="data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )



        # if flag1:
        #     if st.button("Удалить PMI и BRAZIL-WORLD (Низкая корреляция)"):
        #         columns_to_exclude = ['PMI', 'BRAZIL-WORLD']
        #         df = df.drop(columns=columns_to_exclude)


        # # Сохранение DataFrame в Excel
        # excel_file = BytesIO()
        # with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        #     df.to_excel(writer, index=False, sheet_name='Sheet1')
        # excel_file.seek(0)
        #
        # # Кнопка для скачивания Excel
        # st.download_button(
        #     label="Скачать Excel",
        #     data=excel_file,
        #     file_name="data.xlsx",
        #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        # )

