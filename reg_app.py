import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Не используется явно, но может быть полезен matplotlib
import seaborn as sns
import traceback  # Для отладки

# --- 0. Конфигурация страницы ---
st.set_page_config(page_title="Прогнозирование C5TC v2", layout="wide")


# --- 0. Вспомогательные функции ---
def create_lag_features(df, column_name, lags):
    """Создает лаговые признаки."""
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{column_name}_lag_{lag}'] = df_copy[column_name].shift(lag)
    return df_copy


def create_rolling_features(df, column_name, windows, aggregations=['mean', 'std']):
    """Создает признаки на основе скользящего окна."""
    df_copy = df.copy()
    for window in windows:
        for agg in aggregations:
            df_copy[f'{column_name}_rolling_{agg}_{window}'] = df_copy[column_name].shift(1).rolling(window=window,
                                                                                                     min_periods=1).agg(
                agg)
    return df_copy


def create_time_features_simplified(df):
    """Создает упрощенные временные признаки (год, месяц, день) из индекса."""
    df_copy = df.copy()
    if isinstance(df_copy.index, pd.DatetimeIndex):
        date_series = df_copy.index
        df_copy['year'] = date_series.year
        df_copy['month'] = date_series.month
        df_copy['day'] = date_series.day  # День месяца
        # df_copy['dayofweek'] = date_series.dayofweek # Если нужен день недели
        # df_copy['dayofyear'] = date_series.dayofyear # Если нужен день года
        st.sidebar.success("Временные признаки (год, месяц, день) созданы из индекса.")
    else:
        st.sidebar.warning("Не удалось создать временные признаки: индекс не является DatetimeIndex.")
    return df_copy


def plot_feature_importance(model, feature_names, top_n=15):
    """Отображает важность признаков для модели."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        actual_top_n = min(top_n, len(importances))

        fig, ax = plt.subplots(figsize=(10, max(5, actual_top_n * 0.4)))
        sns.barplot(x=importances[indices][:actual_top_n], y=sorted_feature_names[:actual_top_n], ax=ax,
                    palette="viridis", orient='h')
        ax.set_title(f'Топ-{actual_top_n} важных признаков', fontsize=14)
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_ylabel('Признак', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Эта модель не предоставляет информацию о важности признаков.")


def parse_int_list_from_string(s):
    if not s: return []
    try:
        return sorted(list(set([int(item.strip()) for item in s.split(',') if item.strip()])))
    except ValueError:
        st.sidebar.error(f"Неверный формат для списка чисел: {s}")
        return []


# --- Инициализация Session State ---
default_session_keys = [
    'df_raw', 'df_processed', 'df_featured', 'X_train', 'X_test', 'y_train', 'y_test',
    'X_train_final', 'X_test_final', 'model', 'scaler', 'target_col', 'date_col_name',
    'y_pred', 'future_predictions_df', 'original_features_input_str', 'lags_target_str',
    'windows_target_str', 'time_features_enabled', 'scaling_enabled', 'model_params',
    'selected_features_for_model', 'current_uploaded_file_name'
]
for key in default_session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.original_features_input_str is None:
    st.session_state.original_features_input_str = 'brent_oil, iron_ore_price, s&p_500, c10-c14, dollar_index, руда_австралия_сред, руда_бразилия_сред, уголь_австралия_сред'
if st.session_state.lags_target_str is None:
    st.session_state.lags_target_str = "7, 14, 30"
if st.session_state.windows_target_str is None:
    st.session_state.windows_target_str = "15, 30"
if st.session_state.time_features_enabled is None:
    st.session_state.time_features_enabled = True
if st.session_state.scaling_enabled is None:
    st.session_state.scaling_enabled = True
if st.session_state.model_params is None:
    st.session_state.model_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
                                     'min_samples_leaf': 3}

# --- Заголовок приложения ---
st.title("Анализ и Прогнозирование C5TC v2")

# --- Боковая панель для настроек ---
st.sidebar.header("⚙️ Настройки и Параметры")

# --- Шаг 1: Загрузка данных ---
st.sidebar.subheader("1. Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл (.xlsx, .xls)", type=["xlsx", "xls"])

if uploaded_file:
    if st.session_state.get('current_uploaded_file_name') != uploaded_file.name:
        for key in default_session_keys: st.session_state[key] = None
        st.session_state.original_features_input_str = 'brent_oil, iron_ore_price, s&p_500, c10-c14, dollar_index, руда_австралия_сред, руда_бразилия_сред, уголь_австралия_сред'
        st.session_state.lags_target_str = "7, 14, 30"
        st.session_state.windows_target_str = "15, 30"
        st.session_state.time_features_enabled = True
        st.session_state.scaling_enabled = True
        st.session_state.model_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
                                         'min_samples_leaf': 3}
        st.session_state.current_uploaded_file_name = uploaded_file.name

    try:
        st.session_state.df_raw = pd.read_excel(uploaded_file)
        st.success(f"Файл '{uploaded_file.name}' успешно загружен!")
        with st.expander("Предпросмотр загруженных данных (первые 5 строк)", expanded=False):
            st.dataframe(st.session_state.df_raw.head())
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        st.session_state.df_raw = None
else:
    st.info("Пожалуйста, загрузите Excel файл для начала анализа.")
    st.stop()

# --- Шаг 2: Предобработка данных ---
if st.session_state.df_raw is not None:
    st.sidebar.subheader("2. Предобработка")
    df_pp = st.session_state.df_raw.copy()
    df_pp.columns = df_pp.columns.str.replace(' ', '_').str.lower().str.strip()

    DATE_COLUMN_DEFAULT_NAME = 'месяц'  # Имя столбца с датой в исходном файле
    st.session_state.target_col = 'c5tc'

    if DATE_COLUMN_DEFAULT_NAME in df_pp.columns:
        st.session_state.date_col_name = DATE_COLUMN_DEFAULT_NAME  # Сохраняем имя для информации
        st.sidebar.info(f"Столбец с датой для индекса: '{st.session_state.date_col_name}'")
    else:
        st.sidebar.error(f"Столбец '{DATE_COLUMN_DEFAULT_NAME}' (ожидаемый для даты) не найден. Проверьте файл.")
        st.error(f"Не найден столбец '{DATE_COLUMN_DEFAULT_NAME}' для даты.")
        st.stop()

    st.session_state.original_features_input_str = st.sidebar.text_area(
        "Исходные числовые признаки (через запятую):",
        value=st.session_state.original_features_input_str,
        height=100,
        help="Эти признаки (и целевая) будут проверены на NaN до генерации лагов."
    )

    if st.sidebar.button("Применить предобработку", key="preprocess_btn"):
        try:
            df_pp[st.session_state.date_col_name] = pd.to_datetime(df_pp[st.session_state.date_col_name],
                                                                   errors='coerce')
            df_pp.dropna(subset=[st.session_state.date_col_name],
                         inplace=True)  # Удаляем строки, где дата не распозналась
            df_pp.sort_values(by=st.session_state.date_col_name, inplace=True)
            df_pp.set_index(st.session_state.date_col_name, inplace=True)  # Устанавливаем дату как индекс

            initial_features_to_check_str = [name.strip() for name in
                                             st.session_state.original_features_input_str.split(',') if name.strip()]
            cols_to_check_nan = [col for col in initial_features_to_check_str if col in df_pp.columns]
            if st.session_state.target_col and st.session_state.target_col in df_pp.columns and st.session_state.target_col not in cols_to_check_nan:
                cols_to_check_nan.append(st.session_state.target_col)

            if cols_to_check_nan:
                initial_rows = len(df_pp)
                df_pp.dropna(subset=cols_to_check_nan, inplace=True)
                st.sidebar.info(
                    f"Удалено {initial_rows - len(df_pp)} строк из-за NaN в ключевых признаках: {', '.join(cols_to_check_nan)}.")
            else:
                st.sidebar.warning("Не указаны/не найдены признаки для проверки на начальные NaN (кроме целевой).")

            st.session_state.df_processed = df_pp.copy()
            st.success("Предобработка завершена. Даты установлены как индекс.")
            with st.expander("Данные после предобработки (первые 5 строк)", expanded=False):
                st.dataframe(st.session_state.df_processed.head())
        except Exception as e:
            st.error(f"Ошибка на этапе предобработки: {e}")
            st.session_state.df_processed = None

# --- Шаг 3: Генерация признаков ---
if st.session_state.df_processed is not None:
    st.sidebar.subheader("3. Генерация признаков")
    target_col_fe = st.session_state.target_col
    df_fe = st.session_state.df_processed.copy()

    st.session_state.time_features_enabled = st.sidebar.checkbox(
        "Создать временные признаки (год, месяц, день) из индекса?", value=st.session_state.time_features_enabled)
    st.sidebar.markdown(f"**Признаки для целевой переменной '{target_col_fe}':**")
    st.session_state.lags_target_str = st.sidebar.text_input("Лаги (через запятую):",
                                                             value=st.session_state.lags_target_str)
    st.session_state.windows_target_str = st.sidebar.text_input("Скользящие окна (среднее, std; через запятую):",
                                                                value=st.session_state.windows_target_str)

    if st.sidebar.button("Сгенерировать признаки", key="generate_features_btn"):
        if st.session_state.time_features_enabled:
            df_fe = create_time_features_simplified(df_fe)

        lags = parse_int_list_from_string(st.session_state.lags_target_str)
        if lags: df_fe = create_lag_features(df_fe, target_col_fe, lags)

        windows = parse_int_list_from_string(st.session_state.windows_target_str)
        if windows: df_fe = create_rolling_features(df_fe, target_col_fe, windows)

        initial_rows_fe = len(df_fe)
        df_fe.dropna(inplace=True)  # Удаляем NaN, появившиеся после генерации лагов/окон
        st.session_state.df_featured = df_fe.copy()
        st.sidebar.info(f"Удалено {initial_rows_fe - len(df_fe)} строк из-за NaN после генерации признаков.")

        if st.session_state.df_featured.empty:
            st.warning("После генерации признаков и удаления NaN не осталось данных.")
        else:
            st.success("Генерация признаков завершена.")
            with st.expander("Данные с новыми признаками (первые 5 строк)", expanded=False):
                st.dataframe(st.session_state.df_featured.head())

# --- Шаг 4: Подготовка к моделированию ---
if st.session_state.df_featured is not None and not st.session_state.df_featured.empty:
    st.header("📈 Моделирование")
    df_model_input = st.session_state.df_featured.copy()
    target_col_model = st.session_state.target_col

    if target_col_model not in df_model_input.columns:
        st.error(f"Целевая переменная '{target_col_model}' не найдена в данных с признаками.")
        st.stop()

    y = df_model_input[target_col_model]
    potential_X_cols = [col for col in df_model_input.columns if col != target_col_model]

    default_features_selection = potential_X_cols
    if st.session_state.selected_features_for_model:  # Если есть сохраненный выбор
        valid_saved_features = [f for f in st.session_state.selected_features_for_model if f in potential_X_cols]
        if valid_saved_features:
            default_features_selection = valid_saved_features

    with st.expander("4.1 Выбор признаков для модели (X)"):
        st.session_state.selected_features_for_model = st.multiselect(
            "Выберите признаки:", options=potential_X_cols, default=default_features_selection
        )
        if not st.session_state.selected_features_for_model:
            st.warning("Выберите хотя бы один признак для модели.")
            st.stop()
        X = df_model_input[st.session_state.selected_features_for_model]

    col_split, col_scale = st.columns(2)
    with col_split:
        st.subheader("4.2 Разделение данных")
        split_ratio = st.slider("Доля обучающей выборки:", 0.1, 0.9, 0.8, 0.05, key="split_ratio_model")
        split_index = int(len(X) * split_ratio)
        st.session_state.X_train, st.session_state.X_test = X.iloc[:split_index], X.iloc[split_index:]
        st.session_state.y_train, st.session_state.y_test = y.iloc[:split_index], y.iloc[split_index:]
        st.write(f"Обучение: {len(st.session_state.X_train)} строк, Тест: {len(st.session_state.X_test)} строк.")
        if not st.session_state.X_train.empty: st.write(
            f"Обучение до: {st.session_state.X_train.index.max().strftime('%Y-%m-%d')}")
        if not st.session_state.X_test.empty: st.write(
            f"Тест с: {st.session_state.X_test.index.min().strftime('%Y-%m-%d')}")

    with col_scale:
        st.subheader("4.3 Масштабирование")
        st.session_state.scaling_enabled = st.checkbox("Применить StandardScaler к числовым признакам?",
                                                       value=st.session_state.scaling_enabled)
        if st.session_state.X_train.empty or st.session_state.X_test.empty:
            st.warning("Нет данных для масштабирования (обучающая или тестовая выборка пуста).")
            st.session_state.X_train_final, st.session_state.X_test_final = st.session_state.X_train.copy(), st.session_state.X_test.copy()
            st.session_state.scaler = None
        else:
            if st.session_state.scaling_enabled:
                st.session_state.scaler = StandardScaler()
                num_cols_train = st.session_state.X_train.select_dtypes(include=np.number).columns
                num_cols_test = st.session_state.X_test.select_dtypes(include=np.number).columns

                st.session_state.X_train_final = st.session_state.X_train.copy()
                st.session_state.X_test_final = st.session_state.X_test.copy()

                if len(num_cols_train) > 0:
                    st.session_state.X_train_final[num_cols_train] = st.session_state.scaler.fit_transform(
                        st.session_state.X_train[num_cols_train])
                    if len(num_cols_test) > 0:  # Масштабируем тест, только если scaler был обучен
                        st.session_state.X_test_final[num_cols_test] = st.session_state.scaler.transform(
                            st.session_state.X_test[num_cols_test])
                    st.info("StandardScaler применен к числовым признакам.")
                else:
                    st.info("В обучающей выборке нет числовых признаков для масштабирования. Scaler не применялся.")
                    st.session_state.scaler = None  # Убедимся, что scaler сброшен
            else:
                st.session_state.X_train_final, st.session_state.X_test_final = st.session_state.X_train.copy(), st.session_state.X_test.copy()
                st.session_state.scaler = None
                st.info("Масштабирование не применяется.")

# --- Шаг 5: Обучение модели и Оценка ---
if 'X_train_final' in st.session_state and st.session_state.X_train_final is not None and not st.session_state.X_train_final.empty:
    st.subheader("5. Обучение модели RandomForestRegressor")
    m_params = st.session_state.model_params
    n_est = st.slider("Количество деревьев (n_estimators):", 50, 500, m_params.get('n_estimators', 100), 10,
                      key="rf_n_est")
    m_depth = st.slider("Макс. глубина (max_depth):", 3, 30, m_params.get('max_depth', 10), 1, key="rf_m_depth")
    m_split = st.slider("Мин. для разделения (min_samples_split):", 2, 20, m_params.get('min_samples_split', 5), 1,
                        key="rf_m_split")
    m_leaf = st.slider("Мин. в листе (min_samples_leaf):", 1, 20, m_params.get('min_samples_leaf', 3), 1,
                       key="rf_m_leaf")
    st.session_state.model_params = {'n_estimators': n_est, 'max_depth': m_depth, 'min_samples_split': m_split,
                                     'min_samples_leaf': m_leaf}

    if st.button("🚀 Обучить модель и Показать результаты", key="train_model_btn"):
        st.session_state.y_pred = None  # Сброс предыдущих предсказаний на тесте
        st.session_state.future_predictions_df = None  # Сброс предыдущих прогнозов на будущее
        model = RandomForestRegressor(
            n_estimators=n_est, max_depth=m_depth, min_samples_split=m_split,
            min_samples_leaf=m_leaf, random_state=42, n_jobs=-1, max_features='sqrt'
        )
        with st.spinner("Обучение модели..."):
            try:
                model.fit(st.session_state.X_train_final, st.session_state.y_train)
                st.session_state.model = model
                st.success(f"Модель {model.__class__.__name__} обучена!")
            except Exception as e:
                st.error(f"Ошибка при обучении: {e}")
                st.session_state.model = None

        if st.session_state.model:  # Оценка происходит только если модель успешно обучена
            if st.session_state.X_test_final.empty:
                st.warning("Тестовая выборка пуста, оценка невозможна.")
            else:
                try:
                    st.session_state.y_pred = st.session_state.model.predict(st.session_state.X_test_final)
                    mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(st.session_state.y_test, st.session_state.y_pred)
                    r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)

                    st.subheader("🎯 Оценка модели на тестовой выборке")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("R² (Коэфф. детерминации)", f"{r2:.4f}")
                    col_m2.metric("MAE (Ср. абс. ошибка)", f"{mae:.2f}")
                    col_m3.metric("RMSE (Корень из MSE)", f"{rmse:.2f}")
                except Exception as e:
                    st.error(f"Ошибка при оценке модели: {e}")
                    st.session_state.y_pred = None  # Сброс в случае ошибки

# --- ВИЗУАЛИЗАЦИЯ ПОСЛЕ ОБУЧЕНИЯ И ОЦЕНКИ (если модель обучена) ---
if st.session_state.model:
    st.subheader("📊 Визуализация результатов")
    col_v1, col_v2 = st.columns([3, 2])  # Даем больше места графику

    with col_v1:
        st.markdown("**Реальные vs. Предсказанные значения**")
        fig1, ax1 = plt.subplots(figsize=(15, 7))  # Уменьшил немного для лучшего вписывания
        if not st.session_state.y_train.empty:
            ax1.plot(st.session_state.y_train.index, st.session_state.y_train, label='Обучение (Реальные)',
                     color='gray', alpha=0.7, linewidth=1.5)
        if not st.session_state.y_test.empty:
            ax1.plot(st.session_state.y_test.index, st.session_state.y_test, label='Тест (Реальные)', color='blue',
                     marker='.', markersize=8, linewidth=1.5)
        if st.session_state.y_pred is not None and not st.session_state.y_test.empty:
            ax1.plot(st.session_state.y_test.index, st.session_state.y_pred, label='Предсказания (Тест)',
                     color='orange', linestyle='--', marker='x', markersize=6, linewidth=1.5)

        if st.session_state.future_predictions_df is not None and not st.session_state.future_predictions_df.empty:
            ax1.plot(st.session_state.future_predictions_df.index,
                     st.session_state.future_predictions_df[st.session_state.target_col],
                     label='Прогноз на будущее', color='green', linestyle=':', marker='P', markersize=8, linewidth=2)

        ax1.set_xlabel("Дата", fontsize=12);
        ax1.set_ylabel(st.session_state.target_col, fontsize=12)
        ax1.legend(fontsize=10);
        ax1.grid(True, linestyle='--', alpha=0.6);
        fig1.autofmt_xdate()
        ax1.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_v2:
        st.markdown("**Важность признаков**")
        if st.session_state.X_train_final is not None and not st.session_state.X_train_final.empty:
            plot_feature_importance(st.session_state.model, st.session_state.X_train_final.columns)
        else:
            st.info("Нет данных для отображения важности признаков (X_train_final пуст).")

    if st.session_state.y_pred is not None and not st.session_state.y_test.empty:
        with st.expander("График остатков (Тест)", expanded=False):
            residuals = st.session_state.y_test - st.session_state.y_pred
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(residuals.index, residuals, marker='o', linestyle='None', alpha=0.6, color='purple', markersize=5)
            ax2.hlines(0, xmin=residuals.index.min(), xmax=residuals.index.max(), colors='red', linestyles='--')
            ax2.set_xlabel("Дата");
            ax2.set_ylabel("Остатки");
            ax2.set_title("Остатки модели на тестовой выборке")
            ax2.grid(True, linestyle='--', alpha=0.6);
            fig2.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig2)

# --- Шаг 6: Прогноз на будущее ---
if st.session_state.model and st.session_state.df_featured is not None:
    st.header("🔮 Прогноз на будущее")

    num_future_steps = st.number_input("Количество дней для прогноза:", min_value=1, value=7, step=1,
                                       key="future_steps_input")
    # Частота жестко задана как 'D' (ежедневно)
    FUTURE_FREQ = 'D'
    st.info(f"Прогноз будет генерироваться с ежедневной частотой ('{FUTURE_FREQ}').")

    if st.button("Сделать прогноз на будущее", key="predict_future_btn"):
        with st.spinner("Генерация прогноза на будущее..."):
            try:
                model = st.session_state.model
                df_history_for_future = st.session_state.df_featured.copy()
                target_col = st.session_state.target_col
                selected_model_features = list(st.session_state.X_train_final.columns)
                scaler = st.session_state.scaler
                lags_config = parse_int_list_from_string(st.session_state.lags_target_str)
                windows_config = parse_int_list_from_string(st.session_state.windows_target_str)
                time_enabled = st.session_state.time_features_enabled

                generated_target_feature_prefixes = [f"{target_col}_lag_", f"{target_col}_rolling_"]
                time_feature_names = ['year', 'month',
                                      'day']  # Убедитесь, что совпадает с create_time_features_simplified

                truly_exogenous_model_cols = []
                for feat in selected_model_features:
                    is_target_derived = any(feat.startswith(p) for p in generated_target_feature_prefixes)
                    is_time_feat = feat in time_feature_names
                    if not is_target_derived and not is_time_feat:
                        truly_exogenous_model_cols.append(feat)

                if truly_exogenous_model_cols:
                    st.info(
                        f"Для прогноза на будущее, следующие внешние признаки (если используются моделью) будут сохранены на уровне последнего известного значения: {', '.join(truly_exogenous_model_cols)}")

                future_predictions_collector = []
                current_data_for_prediction = df_history_for_future.copy()
                last_date = current_data_for_prediction.index.max()

                for step in range(num_future_steps):
                    next_date = last_date + pd.tseries.frequencies.to_offset(FUTURE_FREQ)
                    new_feature_row = pd.Series(index=current_data_for_prediction.columns, name=next_date,
                                                dtype='float64')
                    last_known_row_overall = current_data_for_prediction.iloc[-1]

                    # 1. Заполняем все не-целевые столбцы из current_data_for_prediction последними значениями
                    for col_name in current_data_for_prediction.columns:
                        if col_name != target_col:  # Целевую будем предсказывать
                            new_feature_row[col_name] = last_known_row_overall[col_name]

                    # 2. Обновляем экзогенные признаки модели (если они есть и не были уже скопированы)
                    for col in truly_exogenous_model_cols:
                        new_feature_row[col] = last_known_row_overall[
                            col]  # Перезапишет, если col есть в truly_exogenous

                    # 3. Генерируем временные признаки для next_date
                    if time_enabled:
                        # Убедимся, что эти столбцы существуют в new_feature_row (были в df_featured) или создаем их
                        if 'year' in new_feature_row.index or 'year' in selected_model_features: new_feature_row[
                            'year'] = next_date.year
                        if 'month' in new_feature_row.index or 'month' in selected_model_features: new_feature_row[
                            'month'] = next_date.month
                        if 'day' in new_feature_row.index or 'day' in selected_model_features: new_feature_row[
                            'day'] = next_date.day

                    # 4. Генерируем лаги для target_col, используя current_data_for_prediction[target_col]
                    # (которая содержит предыдущие предсказанные значения)
                    temp_target_series = current_data_for_prediction[target_col]
                    for lag in lags_config:
                        lag_feat_name = f"{target_col}_lag_{lag}"
                        if lag_feat_name in new_feature_row.index or lag_feat_name in selected_model_features:
                            if len(temp_target_series) >= lag:
                                new_feature_row[lag_feat_name] = temp_target_series.iloc[-lag]
                            else:
                                new_feature_row[lag_feat_name] = np.nan  # Недостаточно истории

                    # 5. Генерируем скользящие окна для target_col
                    for window in windows_config:
                        for agg_func_name in ['mean', 'std']:
                            roll_feat_name = f'{target_col}_rolling_{agg_func_name}_{window}'
                            if roll_feat_name in new_feature_row.index or roll_feat_name in selected_model_features:
                                if len(temp_target_series) >= 1:
                                    # .shift(0) не нужен, т.к. мы берем .iloc[-1] из результата роллинга по temp_target_series
                                    val = \
                                    temp_target_series.rolling(window=window, min_periods=1).agg(agg_func_name).iloc[-1]
                                    new_feature_row[roll_feat_name] = val
                                else:
                                    new_feature_row[roll_feat_name] = np.nan

                    # 6. Собираем X_future только из нужных для модели признаков
                    X_future_step_df = pd.DataFrame([new_feature_row[selected_model_features]],
                                                    columns=selected_model_features, index=[next_date])

                    # 7. Обработка NaN в X_future_step_df
                    if X_future_step_df.isnull().any().any():
                        last_valid_model_features = current_data_for_prediction[selected_model_features].iloc[-1]
                        for col_idx, col_name_fill in enumerate(X_future_step_df.columns):
                            if pd.isnull(X_future_step_df.iloc[0, col_idx]):
                                X_future_step_df.iloc[0, col_idx] = last_valid_model_features[col_name_fill]
                        if X_future_step_df.isnull().any().any():  # Если все еще NaN
                            X_future_step_df = X_future_step_df.fillna(0)

                    # 8. Масштабирование
                    X_future_step_scaled_df = X_future_step_df.copy()
                    if scaler and hasattr(scaler, 'mean_') and scaler.mean_ is not None:  # Проверяем, что scaler обучен
                        num_cols_to_scale = X_future_step_scaled_df.select_dtypes(include=np.number).columns
                        if len(num_cols_to_scale) > 0:
                            X_future_step_scaled_df[num_cols_to_scale] = scaler.transform(
                                X_future_step_scaled_df[num_cols_to_scale])

                    # 9. Предсказание
                    prediction = model.predict(X_future_step_scaled_df)[0]
                    future_predictions_collector.append({'date': next_date, target_col: prediction})

                    # 10. Обновляем current_data_for_prediction для следующей итерации
                    new_feature_row_with_prediction = new_feature_row.copy()
                    new_feature_row_with_prediction[target_col] = prediction  # Записываем предсказание

                    # Добавляем полностью сформированную строку (включая предсказанную цель)
                    current_data_for_prediction = pd.concat([current_data_for_prediction,
                                                             new_feature_row_with_prediction.to_frame().T.astype(
                                                                 current_data_for_prediction.dtypes)],
                                                            ignore_index=False)
                    current_data_for_prediction.index.name = df_history_for_future.index.name  # Восстанавливаем имя индекса
                    last_date = next_date

                st.session_state.future_predictions_df = pd.DataFrame(future_predictions_collector).set_index('date')
                st.success(f"Прогноз на {num_future_steps} дней вперед сгенерирован.")

                with st.expander("Посмотреть данные прогноза на будущее"):
                    st.dataframe(st.session_state.future_predictions_df)

            except Exception as e:
                st.error(f"Ошибка при прогнозировании на будущее: {e}")
                st.error(traceback.format_exc())
                st.session_state.future_predictions_df = None

# --- Подвал ---
st.sidebar.markdown("---")
st.sidebar.info("Анализ C5TC v2")
