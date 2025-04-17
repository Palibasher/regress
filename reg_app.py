import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Анализ C5TC")

st.title("📊 Пошаговый анализ и модель по C5TC")

# --- Загрузка файла ---
uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xlsx"])
if uploaded_file:

    try:
        df = pd.read_excel(uploaded_file)
        st.success("Файл успешно загружен!")

        # Преобразование даты
        df['Месяц'] = pd.to_datetime(df['Месяц'])

        # Удаляем строки с пропущенным C5TC
        df = df.dropna(subset=["C5TC"])

        # Временные признаки
        df['Год'] = df['Месяц'].dt.year
        df['Месяц_числом'] = df['Месяц'].dt.month
        df['День'] = df['Месяц'].dt.day

        df_f = df.drop(columns=['Месяц'])

        st.session_state['df_raw'] = df.copy()
        st.session_state['df_f'] = df_f.copy()

        st.write("📋 Первые строки таблицы:")
        st.dataframe(df_f.head())

        st.success("Дата и временные признаки успешно обработаны.")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")


# --- Шаг 2: Фильтрация выбросов (IQR метод) ---
st.markdown("### 🔍 Шаг 2: Фильтрация выбросов (IQR метод)")

df_f = st.session_state.get('df_f')

if df_f is not None:
    numeric_cols = df_f.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Выберите столбец для анализа выбросов", numeric_cols,
                                index=numeric_cols.index("C5TC") if "C5TC" in numeric_cols else 0)
    iqr_multiplier = st.slider("Множитель IQR", 1.0, 3.0, 1.5, 0.1, key="iqr_slider")

    # IQR анализ по выбранному столбцу
    Q1 = df_f[selected_col].quantile(0.25)
    Q3 = df_f[selected_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    outliers = df_f[(df_f[selected_col] < lower_bound) | (df_f[selected_col] > upper_bound)]

    st.write(f"Обнаружено выбросов в {selected_col}: **{len(outliers)}**")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(y=df_f[selected_col], ax=ax)
    ax.set_title(f"Boxplot: {selected_col}")
    st.pyplot(fig)

    # Очистка всего датафрейма по IQR
    Q1_all = df_f[numeric_cols].quantile(0.25)
    Q3_all = df_f[numeric_cols].quantile(0.75)
    IQR_all = Q3_all - Q1_all
    lower_all = Q1_all - iqr_multiplier * IQR_all
    upper_all = Q3_all + iqr_multiplier * IQR_all

    mask = ~((df_f[numeric_cols] < lower_all) | (df_f[numeric_cols] > upper_all)).any(axis=1)
    removed_data = df_f[~mask]
    cleaned_df = df_f[mask]

    if st.button("Очистить датафрейм от выбросов (IQR)", key="iqr_clean"):
        st.session_state.removed_data = removed_data
        st.session_state.num_removed = len(df_f) - len(cleaned_df)
        st.session_state.df_filtered = cleaned_df.copy()
        st.success(f"Удалено строк с выбросами: {st.session_state.num_removed}")
        with st.expander("📄 Удалённые строки"):
            st.dataframe(st.session_state.removed_data)

# --- Шаг 3: Корреляционный анализ ---
st.markdown("### 🔬 Шаг 3: Корреляционный анализ признаков")

df_filtered = st.session_state.get('df_filtered')

if df_filtered is not None:
    corr_matrix = df_filtered.corr(numeric_only=True)
    show_only_c5tc = st.checkbox("Показать только корреляции с C5TC", value=True)

    if show_only_c5tc and "C5TC" in corr_matrix.columns:
        c5_corr = corr_matrix["C5TC"].drop("C5TC").sort_values(key=lambda x: abs(x), ascending=False)
        st.write("📊 **Корреляции признаков с C5TC:**")
        st.dataframe(c5_corr.to_frame(name="Корреляция").style.background_gradient(cmap='coolwarm', axis=0))
    else:
        st.write("📈 Полная корреляционная матрица:")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
        st.pyplot(fig)

    st.markdown(
        "🔍 **Обратите внимание на признаки, наиболее коррелирующие с C5TC, и возможные признаки с сильной взаимной корреляцией.**")

st.markdown("### 🧪 Шаг 4: Отбор признаков для модели")

df_filtered = st.session_state.get('df_filtered')

if df_filtered is not None:
    # Вычислим корреляции с C5TC
    corr_matrix = df_filtered.corr(numeric_only=True)
    corr_with_c5tc = corr_matrix["C5TC"].drop("C5TC").sort_values(key=lambda x: abs(x), ascending=False)

    # Установим порог для выявления сильно коррелирующих признаков (например, 0.8)
    correlation_threshold = 0.8
    high_corr_pairs = []

    # Ищем пары признаков с высокой корреляцией
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix[col1][col2]) > correlation_threshold:
                high_corr_pairs.append((col1, col2, corr_matrix[col1][col2]))

    # Убираем дублирующиеся пары (например, (A, B) и (B, A))
    high_corr_pairs = sorted(set(tuple(sorted(pair[:2])) + (pair[2],) for pair in high_corr_pairs), key=lambda x: x[2],
                             reverse=True)

    # Выводим список сильных корреляций
    if high_corr_pairs:
        st.markdown(
            "⚠️ Обнаружены пары признаков с высокой корреляцией. Мы рекомендуем выбрать только один из признаков в каждой паре:")
        for pair in high_corr_pairs:
            st.write(f"- Признаки **{pair[0]}** и **{pair[1]}** имеют корреляцию: **{pair[2]:.2f}**")
    else:
        st.success("Не найдено сильно коррелирующих признаков (с корреляцией > 0.8).")

    st.markdown("### Выберите признаки для модели")

    # Создаём два столбца для выбора признаков
    col1, col2 = st.columns(2)

    selected_features = []

    # Перебираем признаки с корреляцией с C5TC
    with col1:
        for i, (feature, corr_value) in enumerate(corr_with_c5tc.items()):
            if i % 2 == 0:  # Левый столбец
                checkbox_label = f"{feature} ({corr_value:.2f})"
                if st.checkbox(checkbox_label, value=True, key=f"feature_{feature}"):
                    selected_features.append(feature)

    with col2:
        for i, (feature, corr_value) in enumerate(corr_with_c5tc.items()):
            if i % 2 != 0:  # Правый столбец
                checkbox_label = f"{feature} ({corr_value:.2f})"
                if st.checkbox(checkbox_label, value=True, key=f"feature_{feature}"):
                    selected_features.append(feature)

    if selected_features:
        st.success(f"Выбрано признаков: {len(selected_features)}")
        st.write("📌 Эти признаки будут использоваться в модели:")
        st.code(", ".join(selected_features))

        # Обновим session_state
        st.session_state.selected_features = selected_features
    else:
        st.warning("❗ Выберите хотя бы один признак для построения модели.")

# --- Шаг 5: Обработка пропусков ---
st.markdown("### 🧹 Шаг 5: Обработка пропусков")

# Получаем выбранные признаки
selected_features = st.session_state.get('selected_features')

if selected_features:
    # Фильтруем датафрейм по выбранным признакам
    df_selected = df_filtered[selected_features]

    # Проверяем наличие пропусков
    missing_data = df_selected.isnull().sum()

    # Покажем информацию о пропусках
    st.write("📊 Количество пропусков в каждом признаке:")
    st.write(missing_data[missing_data > 0])

    # Если есть пропуски, предлагаю обработать их
    if missing_data.any():
        st.markdown("🔧 Как вы хотите обработать пропуски?")

        # Опции для пользователя
        option = st.radio(
            "Выберите метод обработки пропусков:",
            ("Удалить строки с пропусками", "Заполнить пропуски медианой")
        )

        if option == "Удалить строки с пропусками":
            df_cleaned = df_selected.dropna()
            st.session_state.df_cleaned = df_cleaned
            st.success(f"Удалено {len(df_selected) - len(df_cleaned)} строк с пропусками.")

        elif option == "Заполнить пропуски медианой":
            df_imputed = df_selected.fillna(df_selected.median())
            st.session_state.df_cleaned = df_imputed
            st.success("Пропуски успешно заполнены медианой.")

    else:
        st.success("Нет пропусков в выбранных признаках.")
else:
    st.warning("❗ Пожалуйста, выберите признаки для обработки.")

# --- Масштабирование / Нормализация ---
st.markdown("### ⚙️ Предобработка признаков")
if "selected_features" not in st.session_state or not st.session_state.selected_features:
    st.warning("⚠️ Сначала выберите признаки для масштабирования.")
    st.stop()

if "df_cleaned" not in st.session_state:
    st.warning("⚠️ Сначала выполните очистку выбросов или обработку пропусков.")
    st.stop()

use_scaling = st.checkbox("🔄 Применить масштабирование/нормализацию к признакам", value=False)

scaling_method = None
if use_scaling:
    scaling_method = st.radio(
        "Выберите метод масштабирования:",
        ("Стандартизация (StandardScaler)", "Нормализация (MinMaxScaler)"),
        help=(
            "• Стандартизация: переводит данные к распределению с нулевым средним и единичным отклонением.\n"
            "• Нормализация: масштабирует данные в диапазон [0, 1]."
        )
    )

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Используем очищенный датафрейм, если он есть
    df_for_scaling = st.session_state.get("df_cleaned", df_selected).copy()
    scaler = StandardScaler() if scaling_method == "Стандартизация (StandardScaler)" else MinMaxScaler()
    df_for_scaling[selected_features] = scaler.fit_transform(df_for_scaling[selected_features])

    st.session_state.df_scaled = df_for_scaling
    st.success(f"✅ Признаки масштабированы с помощью: {scaling_method}")
    with st.expander("📄 Показать конечный формат"):
        st.dataframe(df_for_scaling.head())
else:
    # Без масштабирования тоже используем очищенный датафрейм
    st.session_state.df_scaled = st.session_state.get("df_cleaned", df_selected).copy()

# --- Выбор модели и построение ---
st.markdown("### 🧠 Построение модели")

# Выбор модели
model_choice = st.radio("Выберите модель:", ["Линейная регрессия", "Случайный лес", "Градиентный бустинг"], horizontal=True)

# Подготовка данных
df_model = st.session_state.df_scaled.copy()
y = df_filtered["C5TC"].loc[df_model.index]  # Убедимся, что индексы совпадают
X = df_model[selected_features]
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_choice == "Линейная регрессия":
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Вычислим метрики
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    std_error = np.sqrt(mean_squared_error(y_test, y_pred))

    y_mean = np.mean(y_test)
    sst = np.sum((y_test - y_mean) ** 2)
    ssr = np.sum((y_pred - y_mean) ** 2)
    sse = np.sum((y_test - y_pred) ** 2)

    msr = ssr / p
    mse = sse / (n - p - 1)
    f_stat = msr / mse

    # Выводим результаты
    st.subheader("📈 Результаты модели линейной регрессии")
    st.markdown(f"""
        - **R²:** {r2:.3f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **R²** (коэффициент детерминации) — это мера того, насколько хорошо модель объясняет вариацию в данных. Значение R² близкое к 1 указывает на то, что модель хорошо описывает данные, а значение, близкое к 0 — что модель не объясняет данные.
        """)

    st.markdown(f"""
        - **Нормированный R² (Adjusted R²):** {adj_r2:.3f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **Нормированный R²** учитывает количество признаков в модели и наказывает за добавление лишних признаков. Это помогает избежать переобучения модели. Его значение всегда ниже или равно R².
        """)

    st.markdown(f"""
        - **Стандартная ошибка:** {std_error:.3f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **Стандартная ошибка** — это мера разброса ошибок предсказания модели. Чем меньше это значение, тем точнее предсказания модели.
        """)

    st.markdown(f"""
        - **SST (Общая сумма квадратов):** {sst:.2f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **SST** (Total Sum of Squares) — это сумма квадратов отклонений всех фактических наблюдений от средней величины целевой переменной. Это общее отклонение в данных.
        """)

    st.markdown(f"""
        - **SSR (Сумма квадратов регрессии):** {ssr:.2f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **SSR** (Sum of Squares for Regression) — это сумма квадратов отклонений прогнозных значений от средней величины целевой переменной. Чем больше эта сумма, тем лучше модель объясняет данные.
        """)

    st.markdown(f"""
        - **SSE (Сумма квадратов ошибок):** {sse:.2f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **SSE** (Sum of Squared Errors) — это сумма квадратов отклонений реальных значений от предсказанных. Меньшее значение SSE означает, что модель точнее предсказывает данные.
        """)

    st.markdown(f"""
        - **MSR (Среднеквадратичное отклонение регрессии):** {msr:.3f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **MSR** — это среднеквадратичное отклонение для регрессии, то есть SSR, деленная на степени свободы модели (количество признаков). Это показатель того, насколько хорошо модель объясняет данные в среднем.
        """)

    st.markdown(f"""
        - **MSE (Среднеквадратичное отклонение ошибки):** {mse:.3f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **MSE** (Mean Squared Error) — это среднее значение квадрата отклонений предсказанных значений от фактических. Меньшее значение MSE означает, что модель имеет меньшие ошибки в предсказаниях.
        """)

    st.markdown(f"""
        - **F-статистика:** {f_stat:.2f}
        """)
    with st.expander("Что это?"):
        st.write("""
        **F-статистика** — это тест, который оценивает значимость модели. Если F-статистика высокая, это означает, что модель значительно лучше предсказывает, чем случайные данные. Зависит от отношения MSR к MSE.
        """)


elif model_choice == "Случайный лес":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Метрики
    r2 = r2_score(y_test, y_pred)
    std_error = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Вывод
    st.subheader("📈 Результаты модели Случайного леса")
    st.markdown(f"""
    - **R²:** {r2:.3f}
    - **Стандартная ошибка (RMSE):** {std_error:.3f}
    - **Средняя абсолютная ошибка (MAE):** {mae:.3f}
    """)


    # --- 📌 Важность признаков ---
    st.markdown("### 🧠 Важность признаков (Feature Importance)")
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importance_sorted = feature_importance.sort_values(ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    feature_importance_sorted.plot(kind='barh', ax=ax2)
    ax2.set_title("Важность признаков по версии Random Forest")
    st.pyplot(fig2)

elif model_choice == "Градиентный бустинг":
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Вычислим метрики
    r2 = r2_score(y_test, y_pred)
    std_error = np.sqrt(mean_squared_error(y_test, y_pred))

    # Выводим результаты
    st.subheader("📈 Результаты модели Градиентного бустинга")
    st.markdown(f"""
    - **R²:** {r2:.3f}
    - **Стандартная ошибка:** {std_error:.3f}
    """)

# Общий график для всех моделей
st.markdown("📊 <b>График: Фактические vs Предсказанные значения</b>", unsafe_allow_html=True)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Фактические значения")
ax.set_ylabel("Предсказанные значения")
ax.set_title(f"Фактические vs Предсказанные ({model_choice})")
st.pyplot(fig)


import seaborn as sns

with st.expander("📊 Density Plot of Real vs Predicted C5TC"):
    st.markdown("Плотности распределения реальных и предсказанных значений C5TC.")
    fig_kde, ax_kde = plt.subplots(figsize=(8, 4))
    sns.kdeplot(y_test, label="Реальные", ax=ax_kde, fill=True, color='blue', alpha=0.5)
    sns.kdeplot(y_pred, label="Предсказанные", ax=ax_kde, fill=True, color='orange', alpha=0.5)
    ax_kde.set_title("Плотность распределения: Real vs Predicted C5TC")
    ax_kde.legend()
    st.pyplot(fig_kde)
