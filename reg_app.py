import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io  # Для работы с загруженным файлом в памяти


# --- Функции для загрузки данных (модифицированы для работы с загруженным файлом) ---
@st.cache_data  # Кэшируем результат обработки файла
def process_uploaded_excel(uploaded_file):
    historical_df_processed = pd.DataFrame()
    ffa_long_df_processed = pd.DataFrame()
    error_messages = []

    if uploaded_file is not None:
        try:
            # Читаем исторические данные
            xls = pd.ExcelFile(uploaded_file)  # Читаем файл один раз
            if HISTORICAL_SHEET_NAME in xls.sheet_names:
                df_hist_temp = pd.read_excel(xls, sheet_name=HISTORICAL_SHEET_NAME)
                if HISTORICAL_DATE_COL in df_hist_temp.columns and HISTORICAL_VALUE_COL in df_hist_temp.columns:
                    df_hist_temp[HISTORICAL_DATE_COL] = pd.to_datetime(df_hist_temp[HISTORICAL_DATE_COL],
                                                                       errors='coerce', dayfirst=True)
                    historical_df_processed = df_hist_temp.dropna(subset=[HISTORICAL_DATE_COL]).sort_values(
                        by=HISTORICAL_DATE_COL).reset_index(drop=True)
                else:
                    error_messages.append(
                        f"На листе '{HISTORICAL_SHEET_NAME}' не найдены колонки '{HISTORICAL_DATE_COL}' или '{HISTORICAL_VALUE_COL}'.")
            else:
                error_messages.append(f"Лист '{HISTORICAL_SHEET_NAME}' для исторических данных не найден в файле.")

            # Читаем FFA данные
            if FFA_LONG_SHEET_NAME in xls.sheet_names:
                df_ffa_temp = pd.read_excel(xls, sheet_name=FFA_LONG_SHEET_NAME)
                required_ffa_cols = [FFA_ARCHIVE_DATE_COL, FFA_START_DATE_COL, FFA_ROUTE_AVG_COL, FFA_ROUTE_ID_COL]
                missing_ffa_cols = [col for col in required_ffa_cols if col not in df_ffa_temp.columns]
                if not missing_ffa_cols:
                    df_ffa_temp[FFA_ARCHIVE_DATE_COL] = pd.to_datetime(df_ffa_temp[FFA_ARCHIVE_DATE_COL],
                                                                       errors='coerce', dayfirst=True)
                    df_ffa_temp[FFA_START_DATE_COL] = pd.to_datetime(df_ffa_temp[FFA_START_DATE_COL], errors='coerce',
                                                                     dayfirst=True)
                    ffa_long_df_processed = df_ffa_temp.dropna(
                        subset=[FFA_ARCHIVE_DATE_COL, FFA_START_DATE_COL, FFA_ROUTE_AVG_COL]).sort_values(
                        by=[FFA_ARCHIVE_DATE_COL, FFA_START_DATE_COL]).reset_index(drop=True)
                else:
                    error_messages.append(
                        f"На листе '{FFA_LONG_SHEET_NAME}' не найдены колонки: {', '.join(missing_ffa_cols)}.")
            else:
                error_messages.append(f"Лист '{FFA_LONG_SHEET_NAME}' для FFA данных не найден в файле.")

        except Exception as e:
            error_messages.append(f"Ошибка при обработке загруженного Excel файла: {e}")
            # Возвращаем пустые DataFrame в случае серьезной ошибки
            return pd.DataFrame(), pd.DataFrame(), error_messages

    return historical_df_processed, ffa_long_df_processed, error_messages


# --- КОНФИГУРАЦИЯ НАЗВАНИЙ ЛИСТОВ И КОЛОНОК (остается, т.к. скрипт ожидает их) ---
HISTORICAL_SHEET_NAME = 'h_data'
HISTORICAL_DATE_COL = 'Date'
HISTORICAL_VALUE_COL = 'C5TC_Value'

FFA_LONG_SHEET_NAME = 'bfa'
FFA_ARCHIVE_DATE_COL = 'ArchiveDate'
FFA_START_DATE_COL = 'StartDate'
FFA_ROUTE_AVG_COL = 'RouteAverage'
FFA_ROUTE_ID_COL = 'RouteIdentifier'

# --- Основное приложение Streamlit ---
st.set_page_config(layout="wide")
st.title("Анализ исторических данных C5TC и форвардных кривых FFA")

# --- Виджет для загрузки файла ---
uploaded_file = st.sidebar.file_uploader("Загрузите ваш Excel файл (с листами 'h_data' и 'bfa')", type=["xlsx", "xls"])

historical_df = pd.DataFrame()
ffa_long_df = pd.DataFrame()

if uploaded_file is not None:
    historical_df, ffa_long_df, errors = process_uploaded_excel(uploaded_file)
    if errors:
        for error in errors:
            st.error(error)
    if historical_df.empty and ffa_long_df.empty and not errors:  # Если файл загружен, но пуст без ошибок парсинга
        st.warning("Загруженный файл не содержит данных на ожидаемых листах или в ожидаемых колонках.")

else:
    st.info("Пожалуйста, загрузите Excel файл для анализа.")
    st.stop()

if historical_df.empty and ffa_long_df.empty:  # Дополнительная проверка после process_uploaded_excel
    st.stop()

# --- Боковая панель для выбора параметров (только если данные загружены) ---
st.sidebar.header("Параметры отображения")

# Получаем ВСЕ уникальные ArchiveDate для выбора, отсортированные
if not ffa_long_df.empty:
    all_available_archive_dates_dt = sorted(ffa_long_df[FFA_ARCHIVE_DATE_COL].dropna().unique())
    all_available_archive_dates_str = [pd.to_datetime(d).strftime('%d.%m.%Y') for d in all_available_archive_dates_dt]
else:
    all_available_archive_dates_str = []

default_selection_count = 3
selected_archive_dates_str = st.sidebar.multiselect(
    "Выберите ArchiveDate для отображения FFA:",
    options=all_available_archive_dates_str,
    default=all_available_archive_dates_str[:min(default_selection_count, len(all_available_archive_dates_str))]
)

show_text_annotations = st.sidebar.checkbox("Показывать значения цен на маркерах FFA", value=True)
ffa_line_width = st.sidebar.slider("Толщина линий FFA:", 1.0, 5.0, 1.5, 0.1)
ffa_marker_size = st.sidebar.slider("Размер маркеров FFA:", 3, 10, 6, 1)

# --- Построение графика ---
fig = go.Figure()

# 1. Добавляем линию исторических данных C5TC
if not historical_df.empty:
    fig.add_trace(go.Scatter(
        x=historical_df[HISTORICAL_DATE_COL],
        y=historical_df[HISTORICAL_VALUE_COL],
        mode='lines',
        name='Historical C5TC',
        line=dict(color='rgba(0, 0, 255, 0.8)', width=2.5)
    ))
else:
    st.warning("Исторические данные не найдены или не загружены. Линия C5TC не будет отображена.")

# 2. Добавляем выбранные фьючерсные кривые FFA
if not ffa_long_df.empty:
    line_colors = ['rgba(255, 0, 0, 0.9)', 'rgba(0, 128, 0, 0.9)', 'rgba(255, 165, 0, 0.9)',
                   'rgba(128, 0, 128, 0.9)', 'rgba(255, 20, 147, 0.9)', 'rgba(0, 191, 255, 0.9)',
                   'rgba(218, 165, 32, 0.9)']
    color_index = 0

    for ad_str in selected_archive_dates_str:
        try:
            selected_archive_date_dt = pd.to_datetime(ad_str, format='%d.%m.%Y')
        except ValueError:
            # st.warning(f"Неверный формат даты '{ad_str}' при выборе. Пропускаем.") # Можно раскомментировать для отладки
            continue

        current_ffa_data = ffa_long_df[ffa_long_df[FFA_ARCHIVE_DATE_COL] == selected_archive_date_dt].copy()

        if current_ffa_data.empty:
            continue

        current_ffa_data = current_ffa_data.sort_values(by=FFA_START_DATE_COL)

        ffa_dates = list(current_ffa_data[FFA_START_DATE_COL])
        ffa_values = list(current_ffa_data[FFA_ROUTE_AVG_COL])
        # ffa_route_ids = list(current_ffa_data[FFA_ROUTE_ID_COL]) # RouteID больше не нужен для текста

        plot_dates = []
        plot_values = []
        plot_texts = []  # Тексты для аннотаций (только цены)

        historical_spot_row = historical_df[historical_df[
                                                HISTORICAL_DATE_COL] == selected_archive_date_dt] if not historical_df.empty else pd.DataFrame()
        if not historical_spot_row.empty:
            spot_val = historical_spot_row[HISTORICAL_VALUE_COL].iloc[0]
            plot_dates.append(selected_archive_date_dt)
            plot_values.append(spot_val)
            plot_texts.append(f"{int(spot_val)}")  # Подпись для спот-точки - только цена

        for i in range(len(ffa_dates)):
            plot_dates.append(ffa_dates[i])
            plot_values.append(ffa_values[i])
            plot_texts.append(f"{int(ffa_values[i])}")  # Текст - только значение цены

        df_plot = pd.DataFrame({'Date': plot_dates, 'Value': plot_values, 'Text': plot_texts})
        df_plot = df_plot.dropna(subset=['Value'])
        if df_plot.empty: continue

        df_plot = df_plot.sort_values('Date')
        df_plot = df_plot.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)

        archive_date_label = selected_archive_date_dt.strftime('%d.%m.%Y')

        current_line_color = line_colors[color_index % len(line_colors)]
        color_index += 1

        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['Value'],
            mode='lines+markers' + ('+text' if show_text_annotations else ''),
            name=f'FFA {archive_date_label}',
            line=dict(dash='solid', width=ffa_line_width, color=current_line_color),
            marker=dict(size=ffa_marker_size, color=current_line_color, line=dict(width=1, color='DarkSlateGrey')),
            text=df_plot['Text'] if show_text_annotations else None,  # Используем только значения цен
            textposition="top center",
            textfont=dict(size=9, color="black"),
            hoverinfo='x+y+name'  # При наведении показываем дату, значение и имя ряда
        ))
else:
    if uploaded_file is not None:  # Если файл был загружен, но ffa_long_df пуст
        st.warning("FFA данные не найдены или не загружены. Фьючерсные кривые не будут отображены.")

# --- Настройка макета графика ---
fig.update_layout(
    height=700,
    title_text='Исторические C5TC и Форвардные кривые FFA',
    xaxis_title='Дата',
    yaxis_title='Ставка',
    legend_title_text='Данные',
    hovermode="x unified",
)

if not fig.data:  # Если на графике нет ни одного ряда данных
    st.info("Нет данных для отображения на графике. Загрузите файл и/или выберите ArchiveDate.")
else:
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.info("""
**Инструкция:**
1. Загрузите ваш Excel файл. Он должен содержать:
    - Лист 'h_data' с историческими данными (колонки 'Date', 'C5TC_Value').
    - Лист 'bfa' с FFA данными (колонки 'ArchiveDate', 'StartDate', 'RouteAverage', 'RouteIdentifier').
2. Выберите одну или несколько дат архивации (ArchiveDate) для отображения фьючерсных кривых.
3. Используйте опции в боковой панели для настройки отображения.
""")
