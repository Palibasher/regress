import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates # Не используется явно
import seaborn as sns
import traceback

# --- 0. Конфигурация страницы ---
st.set_page_config(page_title="Прогнозирование C5TC (XGBoost)", layout="wide")


# --- 0. Вспомогательные функции ---
def create_lag_features(df, column_name, lags):
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{column_name}_lag_{lag}'] = df_copy[column_name].shift(lag)
    return df_copy


def create_rolling_features(df, column_name, windows, aggregations=['mean', 'std']):
    df_copy = df.copy()
    for window in windows:
        for agg in aggregations:
            df_copy[f'{column_name}_rolling_{agg}_{window}'] = df_copy[column_name].shift(1).rolling(window=window,
                                                                                                     min_periods=1).agg(
                agg)
    return df_copy


def create_time_features_simplified(df):
    df_copy = df.copy()
    if isinstance(df_copy.index, pd.DatetimeIndex):
        date_series = df_copy.index
        df_copy['year'] = date_series.year
        df_copy['month'] = date_series.month
        df_copy['day'] = date_series.day
        st.sidebar.success("Временные признаки (год, месяц, день) созданы из индекса.")
    else:
        st.sidebar.warning("Не удалось создать временные признаки: индекс не DatetimeIndex.")
    return df_copy


def plot_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        actual_top_n = min(top_n, len(importances))
        fig, ax = plt.subplots(figsize=(10, max(5, actual_top_n * 0.4)))
        sns.barplot(x=importances[indices][:actual_top_n], y=sorted_feature_names[:actual_top_n], ax=ax,
                    palette="viridis", orient='h')
        ax.set_title(f'Топ-{actual_top_n} важных признаков', fontsize=14)
        ax.set_xlabel('Важность', fontsize=12);
        ax.set_ylabel('Признак', fontsize=12)
        plt.tight_layout();
        st.pyplot(fig)
    else:
        st.info("Эта модель не предоставляет информацию о важности признаков.")


def parse_int_list_from_string(s):
    if not s: return []
    try:
        return sorted(list(set([int(item.strip()) for item in s.split(',') if item.strip()])))
    except ValueError:
        st.sidebar.error(f"Неверный формат: {s}");
        return []


# --- Инициализация Session State ---
default_session_keys = [
    'df_raw', 'df_processed', 'df_featured', 'X_train', 'X_test', 'y_train', 'y_test',
    'X_train_final', 'X_test_final', 'model', 'scaler', 'target_col', 'date_col_name',
    'y_pred',
    'future_predictions_df', 'original_features_input_str', 'lags_target_str',
    'windows_target_str', 'time_features_enabled', 'scaling_enabled', 'model_params_xgb',
    'selected_features_for_model', 'current_uploaded_file_name',
    'lags_for_features_str', 'selected_exog_features_for_lags'  # НОВОЕ для лагов экзогенных признаков
]
for key in default_session_keys:
    if key not in st.session_state: st.session_state[key] = None

if st.session_state.original_features_input_str is None:
    st.session_state.original_features_input_str = 'brent_oil, iron_ore_price, s&p_500, c10-c14, dollar_index, руда_австралия_сред, руда_бразилия_сред, уголь_австралия_сред'
if st.session_state.lags_target_str is None: st.session_state.lags_target_str = "14, 30"
if st.session_state.windows_target_str is None: st.session_state.windows_target_str = "15, 30"
if st.session_state.time_features_enabled is None: st.session_state.time_features_enabled = True
if st.session_state.scaling_enabled is None: st.session_state.scaling_enabled = True
if st.session_state.model_params_xgb is None:
    st.session_state.model_params_xgb = {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1
    }
if st.session_state.lags_for_features_str is None:  # НОВОЕ
    st.session_state.lags_for_features_str = "1, 3, 7"
if st.session_state.selected_exog_features_for_lags is None:  # НОВОЕ
    st.session_state.selected_exog_features_for_lags = []

# --- Заголовок приложения ---
st.title("Анализ и Прогнозирование C5TC (XGBoost с лагами признаков)")

# --- Боковая панель для настроек ---
st.sidebar.header("⚙️ Настройки и Параметры")

# --- Шаг 1: Загрузка данных ---
st.sidebar.subheader("1. Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл (.xlsx, .xls)", type=["xlsx", "xls"])
if uploaded_file:
    if st.session_state.get('current_uploaded_file_name') != uploaded_file.name:
        for key in default_session_keys: st.session_state[key] = None
        st.session_state.original_features_input_str = 'brent_oil, iron_ore_price, s&p_500, c10-c14, dollar_index, руда_австралия_сред, руда_бразилия_сред, уголь_австралия_сред'
        st.session_state.lags_target_str = "14, 30"
        st.session_state.windows_target_str = "15, 30"
        st.session_state.time_features_enabled = True
        st.session_state.scaling_enabled = True
        st.session_state.model_params_xgb = {
            'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1
        }
        st.session_state.lags_for_features_str = "1, 3, 7"  # НОВОЕ
        st.session_state.selected_exog_features_for_lags = []  # НОВОЕ
        st.session_state.current_uploaded_file_name = uploaded_file.name
    try:
        st.session_state.df_raw = pd.read_excel(uploaded_file)
        st.success(f"Файл '{uploaded_file.name}' успешно загружен!")
        with st.expander("Предпросмотр (первые 5 строк)", expanded=False):
            st.dataframe(st.session_state.df_raw.head())
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}");
        st.session_state.df_raw = None
else:
    st.info("Загрузите Excel файл.");
    st.stop()

# --- Шаг 2: Предобработка данных ---
if st.session_state.df_raw is not None:
    st.sidebar.subheader("2. Предобработка")
    df_pp = st.session_state.df_raw.copy()
    df_pp.columns = df_pp.columns.str.replace(' ', '_').str.lower().str.strip()
    DATE_COLUMN_DEFAULT_NAME = 'месяц';
    st.session_state.target_col = 'c5tc'
    if DATE_COLUMN_DEFAULT_NAME in df_pp.columns:
        st.session_state.date_col_name = DATE_COLUMN_DEFAULT_NAME
    else:
        st.sidebar.error(f"Столбец '{DATE_COLUMN_DEFAULT_NAME}' не найден.");
        st.error("Не найден столбец даты.");
        st.stop()

    st.session_state.original_features_input_str = st.sidebar.text_area(
        "Исходные числовые признаки (через запятую, будут проверены на NaN):",
        value=st.session_state.original_features_input_str, height=100
    )
    if st.sidebar.button("Применить предобработку", key="preprocess_btn"):
        try:
            df_pp[st.session_state.date_col_name] = pd.to_datetime(df_pp[st.session_state.date_col_name],
                                                                   errors='coerce')
            df_pp.dropna(subset=[st.session_state.date_col_name], inplace=True)
            df_pp.sort_values(by=st.session_state.date_col_name, inplace=True)
            df_pp.set_index(st.session_state.date_col_name, inplace=True)

            initial_features_to_check = [name.strip().lower().replace(' ', '_') for name in
                                         st.session_state.original_features_input_str.split(',') if name.strip()]
            cols_to_check_nan = [col for col in initial_features_to_check if col in df_pp.columns]
            if st.session_state.target_col and st.session_state.target_col in df_pp.columns and st.session_state.target_col not in cols_to_check_nan:
                cols_to_check_nan.append(st.session_state.target_col)

            if cols_to_check_nan:
                initial_rows = len(df_pp);
                df_pp.dropna(subset=cols_to_check_nan, inplace=True)
                st.sidebar.info(
                    f"Удалено {initial_rows - len(df_pp)} строк из-за NaN в: {', '.join(cols_to_check_nan)}.")
            st.session_state.df_processed = df_pp.copy();
            st.success("Предобработка завершена.")
            with st.expander("Данные после предобработки (первые 5)", expanded=False):
                st.dataframe(st.session_state.df_processed.head())
        except Exception as e:
            st.error(f"Ошибка предобработки: {e}");
            st.session_state.df_processed = None

# --- Шаг 3: Генерация признаков ---
if st.session_state.df_processed is not None:
    st.sidebar.subheader("3. Генерация признаков")
    target_col_fe = st.session_state.target_col
    df_fe_input = st.session_state.df_processed.copy()  # Используем df_processed как вход

    st.session_state.time_features_enabled = st.sidebar.checkbox("Создать временные признаки (год, месяц, день)?",
                                                                 value=st.session_state.time_features_enabled)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Признаки для целевой переменной '{target_col_fe}':**")
    st.session_state.lags_target_str = st.sidebar.text_input(f"Лаги для '{target_col_fe}' (через запятую):",
                                                             value=st.session_state.lags_target_str)
    st.session_state.windows_target_str = st.sidebar.text_input(
        f"Скользящие окна для '{target_col_fe}' (среднее, std; через запятую):",
        value=st.session_state.windows_target_str)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Лаговые признаки для ИСХОДНЫХ признаков:**")

    # Формируем список потенциальных экзогенных признаков из df_processed
    potential_exog_features = [col for col in df_fe_input.columns if col != target_col_fe]

    # Дефолтные значения для multiselect берем из сохраненного состояния или из original_features_input_str
    default_exog_for_lags = st.session_state.selected_exog_features_for_lags
    if not default_exog_for_lags:  # Если пусто, пытаемся взять из original_features_input_str
        raw_original_features = [name.strip().lower().replace(' ', '_') for name in
                                 st.session_state.original_features_input_str.split(',') if name.strip()]
        default_exog_for_lags = [feat for feat in raw_original_features if
                                 feat in potential_exog_features and feat != target_col_fe]

    st.session_state.selected_exog_features_for_lags = st.sidebar.multiselect(
        "Выберите исходные признаки для создания лагов:",
        options=potential_exog_features,
        default=default_exog_for_lags,
        key="selected_exog_features_for_lags_widget"
    )
    st.session_state.lags_for_features_str = st.sidebar.text_input(
        "Лаги для выбранных исходных признаков (через запятую, одинаковые для всех):",
        value=st.session_state.lags_for_features_str,
        key="lags_for_features_input_widget"
    )

    if st.sidebar.button("Сгенерировать признаки", key="generate_features_btn"):
        df_fe_output = df_fe_input.copy()  # Работаем с копией на этом этапе
        if st.session_state.time_features_enabled:
            df_fe_output = create_time_features_simplified(df_fe_output)

        if st.session_state.lags_target_str:
            lags_target = parse_int_list_from_string(st.session_state.lags_target_str)
            if lags_target: df_fe_output = create_lag_features(df_fe_output, target_col_fe, lags_target)

        if st.session_state.windows_target_str:
            windows_target = parse_int_list_from_string(st.session_state.windows_target_str)
            if windows_target: df_fe_output = create_rolling_features(df_fe_output, target_col_fe, windows_target)

        if st.session_state.selected_exog_features_for_lags and st.session_state.lags_for_features_str:
            lags_for_exog_features = parse_int_list_from_string(st.session_state.lags_for_features_str)
            if lags_for_exog_features:
                for feature_name in st.session_state.selected_exog_features_for_lags:
                    if feature_name in df_fe_output.columns:
                        df_fe_output = create_lag_features(df_fe_output, feature_name, lags_for_exog_features)
                        st.sidebar.info(f"Созданы лаги {lags_for_exog_features} для признака '{feature_name}'.")
                    else:
                        st.sidebar.warning(
                            f"Признак '{feature_name}' не найден в обработанных данных для создания лагов.")
            else:
                st.sidebar.warning("Не указаны или неверный формат лагов для исходных признаков.")

        initial_rows_fe = len(df_fe_output)
        df_fe_output.dropna(inplace=True)
        st.session_state.df_featured = df_fe_output.copy()  # Сохраняем результат в session_state
        st.sidebar.info(
            f"Удалено {initial_rows_fe - len(st.session_state.df_featured)} строк из-за NaN после генерации ВСЕХ признаков.")

        if st.session_state.df_featured.empty:
            st.warning("После генерации признаков и удаления NaN не осталось данных.")
        else:
            st.success("Генерация признаков завершена.");
            with st.expander("Данные с новыми признаками (первые 5 строк)", expanded=False):
                st.dataframe(st.session_state.df_featured.head())

# --- Шаг 4: Подготовка к моделированию ---
if st.session_state.df_featured is not None and not st.session_state.df_featured.empty:
    st.header("📈 Моделирование XGBoost")
    df_model_input = st.session_state.df_featured.copy();
    target_col_model = st.session_state.target_col
    if target_col_model not in df_model_input.columns: st.error(f"Цель '{target_col_model}' не найдена."); st.stop()
    y = df_model_input[target_col_model];

    # potential_X_cols теперь берется из df_featured, где уже есть все сгенерированные признаки
    potential_X_cols = [col for col in df_model_input.columns if col != target_col_model]

    # Обновляем default_features_selection, если есть сохраненные и они валидны
    default_features_selection = potential_X_cols
    if st.session_state.selected_features_for_model:
        valid_saved_features = [f for f in st.session_state.selected_features_for_model if f in potential_X_cols]
        if valid_saved_features: default_features_selection = valid_saved_features

    with st.expander("4.1 Выбор признаков для модели (X)"):
        st.session_state.selected_features_for_model = st.multiselect(
            "Выберите признаки (включая сгенерированные лаги):",
            options=potential_X_cols,
            default=default_features_selection
        )
        if not st.session_state.selected_features_for_model: st.warning("Выберите хотя бы один признак."); st.stop()
        X = df_model_input[st.session_state.selected_features_for_model]

    col_split, col_scale = st.columns(2)
    with col_split:
        st.subheader("4.2 Разделение данных");
        split_ratio = st.slider("Доля обучающей:", 0.1, 0.9, 0.8, 0.05, key="split_ratio_model")
        split_index = int(len(X) * split_ratio)
        st.session_state.X_train, st.session_state.X_test = X.iloc[:split_index], X.iloc[split_index:]
        st.session_state.y_train, st.session_state.y_test = y.iloc[:split_index], y.iloc[split_index:]
        st.write(f"Обучение: {len(st.session_state.X_train)} строк, Тест: {len(st.session_state.X_test)} строк.")
        if not st.session_state.X_train.empty: st.write(f"Обучение до: {st.session_state.X_train.index.max():%Y-%m-%d}")
        if not st.session_state.X_test.empty: st.write(f"Тест с: {st.session_state.X_test.index.min():%Y-%m-%d}")
    with col_scale:
        st.subheader("4.3 Масштабирование");
        st.session_state.scaling_enabled = st.checkbox("Применить StandardScaler?",
                                                       value=st.session_state.scaling_enabled)
        if st.session_state.X_train.empty or st.session_state.X_test.empty:
            st.warning("Нет данных для масштабирования.");
            st.session_state.X_train_final, st.session_state.X_test_final = st.session_state.X_train.copy(), st.session_state.X_test.copy();
            st.session_state.scaler = None
        else:
            if st.session_state.scaling_enabled:
                st.session_state.scaler = StandardScaler();
                num_cols_train = st.session_state.X_train.select_dtypes(include=np.number).columns
                st.session_state.X_train_final = st.session_state.X_train.copy();
                st.session_state.X_test_final = st.session_state.X_test.copy()
                if len(num_cols_train) > 0:
                    # Убедимся, что масштабируем только существующие колонки в X_train_final
                    cols_to_scale_train = [col for col in num_cols_train if
                                           col in st.session_state.X_train_final.columns]
                    if cols_to_scale_train:
                        st.session_state.X_train_final[cols_to_scale_train] = st.session_state.scaler.fit_transform(
                            st.session_state.X_train[cols_to_scale_train])

                    num_cols_test = st.session_state.X_test.select_dtypes(include=np.number).columns
                    # Убедимся, что масштабируем только существующие колонки в X_test_final и те, что были в обучении
                    cols_to_scale_test = [col for col in num_cols_test if
                                          col in st.session_state.X_test_final.columns and col in st.session_state.scaler.feature_names_in_]
                    if cols_to_scale_test:
                        st.session_state.X_test_final[cols_to_scale_test] = st.session_state.scaler.transform(
                            st.session_state.X_test[cols_to_scale_test])
                    st.info("StandardScaler применен.")
                else:
                    st.info("Нет числовых признаков для масштабирования.");
                    st.session_state.scaler = None
            else:
                st.session_state.X_train_final, st.session_state.X_test_final = st.session_state.X_train.copy(), st.session_state.X_test.copy();
                st.session_state.scaler = None;
                st.info("Масштабирование не применяется.")

# --- Шаг 5: Обучение модели XGBoost и Оценка ---
if 'X_train_final' in st.session_state and st.session_state.X_train_final is not None and not st.session_state.X_train_final.empty:
    st.subheader("5. Обучение модели XGBoost")
    m_params_xgb = st.session_state.model_params_xgb

    n_est_xgb = st.slider("Количество деревьев (n_estimators):", 50, 500, m_params_xgb.get('n_estimators', 200), 10,
                          key="xgb_n_est")
    lr_xgb = st.select_slider("Скорость обучения (learning_rate):", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                              value=m_params_xgb.get('learning_rate', 0.05), key="xgb_lr")
    m_depth_xgb = st.slider("Макс. глубина (max_depth):", 2, 10, m_params_xgb.get('max_depth', 4), 1, key="xgb_m_depth")
    subsample_xgb = st.slider("Доля выборок (subsample):", 0.5, 1.0, m_params_xgb.get('subsample', 0.8), 0.1,
                              key="xgb_subsample")
    colsample_xgb = st.slider("Доля признаков для дерева (colsample_bytree):", 0.5, 1.0,
                              m_params_xgb.get('colsample_bytree', 0.8), 0.1, key="xgb_colsample")

    st.session_state.model_params_xgb = {
        'n_estimators': n_est_xgb, 'learning_rate': lr_xgb, 'max_depth': m_depth_xgb,
        'subsample': subsample_xgb, 'colsample_bytree': colsample_xgb,
        'reg_alpha': m_params_xgb.get('reg_alpha', 0.1), 'reg_lambda': m_params_xgb.get('reg_lambda', 0.1)
    }

    if st.button("🚀 Обучить XGBoost и Показать результаты", key="train_xgb_btn"):
        st.session_state.y_pred = None;
        st.session_state.future_predictions_df = None

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_est_xgb, learning_rate=lr_xgb, max_depth=m_depth_xgb,
            subsample=subsample_xgb, colsample_bytree=colsample_xgb,
            reg_alpha=st.session_state.model_params_xgb['reg_alpha'],
            reg_lambda=st.session_state.model_params_xgb['reg_lambda'],
            random_state=42, n_jobs=-1
        )
        with st.spinner("Обучение модели XGBoost..."):
            try:
                model.fit(st.session_state.X_train_final, st.session_state.y_train)
                st.session_state.model = model
                st.success(f"Модель {model.__class__.__name__} обучена!")
                if not st.session_state.X_test_final.empty:
                    st.session_state.y_pred = model.predict(st.session_state.X_test_final)
                else:
                    st.session_state.y_pred = np.array([])
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}");
                st.session_state.model = None;
                st.session_state.y_pred = None

        if st.session_state.model:
            if st.session_state.X_test_final.empty:
                st.warning("Тестовая выборка пуста.")
            elif st.session_state.y_pred is not None and len(st.session_state.y_pred) > 0:
                mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
                rmse = np.sqrt(mse);
                mae = mean_absolute_error(st.session_state.y_test, st.session_state.y_pred)
                r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
                st.subheader("🎯 Оценка модели XGBoost на тестовой выборке")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("R²", f"{r2:.4f}");
                col_m2.metric("MAE", f"{mae:.2f}");
                col_m3.metric("RMSE", f"{rmse:.2f}")
            else:
                st.warning("Нет предсказаний модели для оценки.")

# --- ВИЗУАЛИЗАЦИЯ ПОСЛЕ ОБУЧЕНИЯ ---
if st.session_state.model:
    st.subheader("📊 Визуализация результатов (XGBoost)")
    col_v1, col_v2 = st.columns([3, 2])
    with col_v1:
        st.markdown("**Факт (Тест) vs. Предсказания**")
        fig1, ax1 = plt.subplots(figsize=(15, 7))
        y_test_index = st.session_state.y_test.index if not st.session_state.y_test.empty else None

        if not st.session_state.y_test.empty and y_test_index is not None:
            ax1.plot(y_test_index, st.session_state.y_test, label='Тест (Факт)', color='blue', marker='o', markersize=5,
                     linewidth=1.5, alpha=0.7)
        if st.session_state.y_pred is not None and y_test_index is not None and len(st.session_state.y_pred) == len(
                y_test_index):
            ax1.plot(y_test_index, st.session_state.y_pred, label='Предсказания (Тест)', color='orange', linestyle='-',
                     marker='.', markersize=6, linewidth=1.5)

        # Убрали прогноз на будущее с этого графика
        # if st.session_state.future_predictions_df is not None and not st.session_state.future_predictions_df.empty:
        #     ax1.plot(st.session_state.future_predictions_df.index, st.session_state.future_predictions_df[st.session_state.target_col], ...)

        # Устанавливаем пределы оси X только по тестовым данным, если они есть
        if y_test_index is not None:
            ax1.set_xlim([y_test_index.min(), y_test_index.max()])

        ax1.set_xlabel("Дата", fontsize=12);
        ax1.set_ylabel(st.session_state.target_col, fontsize=12);
        ax1.legend(fontsize=10);
        ax1.grid(True, linestyle='--', alpha=0.6);
        fig1.autofmt_xdate();
        ax1.tick_params(axis='both', which='major', labelsize=10);
        plt.tight_layout();
        st.pyplot(fig1)

    with col_v2:
        st.markdown("**Важность признаков (XGBoost)**")
        if st.session_state.model is not None and st.session_state.X_train_final is not None and not st.session_state.X_train_final.empty:
            plot_feature_importance(st.session_state.model, st.session_state.X_train_final.columns)
        else:
            st.info("Нет данных для важности признаков.")

    if st.session_state.y_pred is not None and not st.session_state.y_test.empty:
        with st.expander("График остатков (Тест XGBoost)", expanded=False):
            residuals = st.session_state.y_test - st.session_state.y_pred;
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(residuals.index, residuals, marker='o', linestyle='None', alpha=0.6, color='purple', markersize=5);
            ax2.hlines(0, xmin=residuals.index.min(), xmax=residuals.index.max(), colors='red', linestyles='--')
            ax2.set_xlabel("Дата");
            ax2.set_ylabel("Остатки");
            ax2.set_title("Остатки XGBoost на тесте");
            ax2.grid(True, linestyle='--', alpha=0.6);
            fig2.autofmt_xdate();
            plt.tight_layout();
            st.pyplot(fig2)

# --- Шаг 6: Прогноз на будущее ---
# (Остается без изменений, но теперь будет ЕДИНСТВЕННЫМ местом отображения прогноза на будущее)
if st.session_state.model and st.session_state.df_featured is not None:
    st.header("🔮 Прогноз на будущее (XGBoost)")
    num_future_steps = st.number_input("Количество дней для прогноза:", min_value=1, value=7, step=1,
                                       key="future_steps_input")
    FUTURE_FREQ = 'D';
    st.info(f"Прогноз с ежедневной частотой ('{FUTURE_FREQ}').")
    if st.button("Сделать прогноз XGBoost на будущее", key="predict_future_xgb_btn"):
        # Очищаем предыдущий прогноз на будущее перед генерацией нового
        st.session_state.future_predictions_df = None
        with st.spinner("Генерация прогноза XGBoost..."):
            try:
                model_to_use_for_future = st.session_state.model
                df_history_for_future = st.session_state.df_featured.copy()  # Берем данные СО ВСЕМИ признаками
                target_col = st.session_state.target_col;
                selected_model_features = list(
                    st.session_state.X_train_final.columns)  # Признаки, на которых обучалась модель
                scaler = st.session_state.scaler;

                # Получаем конфигурацию лагов и окон из session_state (они нужны для генерации признаков на лету)
                lags_target_config = parse_int_list_from_string(st.session_state.lags_target_str)
                windows_target_config = parse_int_list_from_string(st.session_state.windows_target_str)
                lags_exog_config = parse_int_list_from_string(st.session_state.lags_for_features_str)
                selected_exog_for_lags_config = st.session_state.selected_exog_features_for_lags

                time_enabled = st.session_state.time_features_enabled

                future_predictions_collector = [];
                # df_current_for_future_preds будет расширяться на каждом шаге
                df_current_for_future_preds = df_history_for_future.copy()
                last_date = df_current_for_future_preds.index.max()

                for step in range(num_future_steps):
                    next_date = last_date + pd.tseries.frequencies.to_offset(FUTURE_FREQ)

                    # Создаем строку для новых признаков
                    new_feature_row_dict = {}

                    # 1. Экзогенные признаки (не лагированные и не целевая) - берем последние известные значения
                    # Копируем последние известные значения для всех столбцов, которые есть в selected_model_features,
                    # но не являются производными от целевой переменной или временными.
                    last_known_non_derived_row = df_current_for_future_preds.iloc[-1]

                    for feat_name in selected_model_features:
                        is_target_lag = feat_name.startswith(target_col + "_lag_")
                        is_target_roll = feat_name.startswith(target_col + "_rolling_")
                        is_time_feat = feat_name in ['year', 'month', 'day']
                        is_exog_lag = any(
                            feat_name.startswith(ex_feat + "_lag_") for ex_feat in selected_exog_for_lags_config)

                        if not is_target_lag and not is_target_roll and not is_time_feat and not is_exog_lag and feat_name in last_known_non_derived_row.index:
                            new_feature_row_dict[feat_name] = last_known_non_derived_row[feat_name]

                    # 2. Временные признаки
                    if time_enabled:
                        if 'year' in selected_model_features: new_feature_row_dict['year'] = next_date.year
                        if 'month' in selected_model_features: new_feature_row_dict['month'] = next_date.month
                        if 'day' in selected_model_features: new_feature_row_dict['day'] = next_date.day

                    # 3. Лаги и скользящие окна от целевой переменной
                    temp_target_series_for_fe = df_current_for_future_preds[target_col]
                    for lag_t in lags_target_config:
                        lag_feat_name_t = f"{target_col}_lag_{lag_t}"
                        if lag_feat_name_t in selected_model_features:
                            if len(temp_target_series_for_fe) >= lag_t:
                                new_feature_row_dict[lag_feat_name_t] = temp_target_series_for_fe.iloc[-lag_t]
                            else:
                                new_feature_row_dict[lag_feat_name_t] = np.nan
                    for window_t in windows_target_config:
                        for agg_func_name_t in ['mean', 'std']:
                            roll_feat_name_t = f'{target_col}_rolling_{agg_func_name_t}_{window_t}'
                            if roll_feat_name_t in selected_model_features:
                                if len(temp_target_series_for_fe) >= 1:
                                    val_t = temp_target_series_for_fe.rolling(window=window_t, min_periods=1).agg(
                                        agg_func_name_t).iloc[-1]
                                    new_feature_row_dict[roll_feat_name_t] = val_t
                                else:
                                    new_feature_row_dict[roll_feat_name_t] = np.nan

                    # 4. Лаги от экзогенных признаков
                    for exog_feat_name in selected_exog_for_lags_config:
                        if exog_feat_name in df_current_for_future_preds.columns:
                            temp_exog_series = df_current_for_future_preds[exog_feat_name]
                            for lag_e in lags_exog_config:
                                lag_feat_name_e = f"{exog_feat_name}_lag_{lag_e}"
                                if lag_feat_name_e in selected_model_features:
                                    if len(temp_exog_series) >= lag_e:
                                        new_feature_row_dict[lag_feat_name_e] = temp_exog_series.iloc[-lag_e]
                                    else:
                                        new_feature_row_dict[lag_feat_name_e] = np.nan
                        else:  # Если исходный экзогенный признак отсутствует в df_current_for_future_preds (маловероятно, но для безопасности)
                            for lag_e in lags_exog_config:
                                lag_feat_name_e = f"{exog_feat_name}_lag_{lag_e}"
                                if lag_feat_name_e in selected_model_features:
                                    new_feature_row_dict[lag_feat_name_e] = np.nan

                    X_future_step_df_from_dict = {feat: new_feature_row_dict.get(feat, np.nan) for feat in
                                                  selected_model_features}
                    X_future_step_df = pd.DataFrame([X_future_step_df_from_dict], columns=selected_model_features,
                                                    index=[next_date])

                    if X_future_step_df.isnull().any().any():
                        last_valid_model_features_from_hist = df_current_for_future_preds[selected_model_features].iloc[
                            -1]
                        for col_fill in X_future_step_df.columns:
                            if pd.isnull(X_future_step_df.loc[next_date, col_fill]):
                                X_future_step_df.loc[next_date, col_fill] = last_valid_model_features_from_hist.get(
                                    col_fill, 0)  # 0 как крайняя мера
                        if X_future_step_df.isnull().any().any(): X_future_step_df.fillna(0,
                                                                                          inplace=True)  # Еще раз, если get вернул None

                    X_future_step_scaled_df = X_future_step_df.copy()
                    if scaler and hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                        num_cols_to_scale = X_future_step_scaled_df.select_dtypes(include=np.number).columns
                        if len(num_cols_to_scale) > 0:
                            cols_in_scaler = scaler.feature_names_in_ if hasattr(scaler,
                                                                                 'feature_names_in_') else num_cols_to_scale
                            cols_to_scale_now = [col for col in num_cols_to_scale if col in cols_in_scaler]
                            if cols_to_scale_now: X_future_step_scaled_df[cols_to_scale_now] = scaler.transform(
                                X_future_step_scaled_df[cols_to_scale_now])

                    prediction_step = model_to_use_for_future.predict(X_future_step_scaled_df)[0]
                    future_predictions_collector.append({'date': next_date, target_col: prediction_step})

                    # Добавляем предсказанное значение и ВСЕ сгенерированные признаки в историю для следующего шага
                    new_row_for_history = X_future_step_df.iloc[0].copy()  # Копируем все сгенерированные признаки
                    new_row_for_history[target_col] = prediction_step  # Добавляем предсказанную цель

                    # Нужно убедиться, что все колонки из df_current_for_future_preds присутствуют
                    # и имеют правильные типы данных перед конкатенацией.
                    # Копируем исходные экзогенные признаки, если они не были перегенерированы как лаги
                    for original_col_name in df_current_for_future_preds.columns:
                        if original_col_name not in new_row_for_history:
                            # Если это исходный экзогенный признак, значение которого мы должны нести дальше
                            if original_col_name in last_known_non_derived_row.index and original_col_name not in selected_exog_for_lags_config:
                                new_row_for_history[original_col_name] = last_known_non_derived_row[original_col_name]
                            # Иначе (например, старый лаг цели, который больше не нужен), можно оставить NaN или 0
                            # Для простоты, если колонка не была сгенерирована и не является "несущим" экзогеном, оставим как есть (будет NaN, если нет в new_row_for_history)

                    new_row_df_for_concat = pd.DataFrame([new_row_for_history], index=[next_date])
                    new_row_df_for_concat.index.name = df_current_for_future_preds.index.name

                    # Перед concat, выравниваем колонки и типы
                    df_current_for_future_preds = pd.concat([
                        df_current_for_future_preds,
                        new_row_df_for_concat.reindex(columns=df_current_for_future_preds.columns).astype(
                            df_current_for_future_preds.dtypes, errors='ignore')
                    ], ignore_index=False)
                    last_date = next_date  # Обновляем last_date

                st.session_state.future_predictions_df = pd.DataFrame(future_predictions_collector).set_index('date');
                st.success(f"Прогноз XGBoost на {num_future_steps} дней сгенерирован.")

                # --- НОВОЕ: Отдельная визуализация прогноза на будущее ---
                if st.session_state.future_predictions_df is not None and not st.session_state.future_predictions_df.empty:
                    with st.expander("График прогноза на будущее", expanded=True):
                        fig_future, ax_future = plt.subplots(figsize=(12, 6))
                        # Можно добавить немного истории для контекста
                        history_to_plot = st.session_state.y_test.tail(30)  # последние 30 дней теста
                        if not history_to_plot.empty:
                            ax_future.plot(history_to_plot.index, history_to_plot, label='История (Тест, Факт)',
                                           color='gray', alpha=0.7)

                        ax_future.plot(st.session_state.future_predictions_df.index,
                                       st.session_state.future_predictions_df[target_col],
                                       label='Прогноз на будущее', color='purple', marker='o')
                        ax_future.set_title('Прогноз на будущее')
                        ax_future.set_xlabel('Дата')
                        ax_future.set_ylabel(target_col)
                        ax_future.legend()
                        ax_future.grid(True)
                        fig_future.autofmt_xdate()
                        st.pyplot(fig_future)
                    with st.expander("Данные прогноза XGBoost на будущее"):
                        st.dataframe(st.session_state.future_predictions_df)
                # -------------------------------------------------------
            except Exception as e:
                st.error(f"Ошибка прогноза XGBoost: {e}");
                st.error(traceback.format_exc());
                st.session_state.future_predictions_df = None

# --- Подвал ---
st.sidebar.markdown("---");
st.sidebar.info("Анализ C5TC (XGBoost с лагами признаков)")
