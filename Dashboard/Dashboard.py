import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Analisis Data Rental Sepeda",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load dan proses data
@st.cache_data
def load_data():
    main_data = pd.read_csv("Dashboard/main_data.csv")
    main_data['dteday'] = pd.to_datetime(main_data['dteday'])

    # Tambah kolom tanggal untuk filtering
    main_data['year'] = main_data['dteday'].dt.year
    main_data['month'] = main_data['dteday'].dt.month
    main_data['day'] = main_data['dteday'].dt.day
    main_data['date'] = main_data['dteday'].dt.date

    # Season mapping
    season_labels = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    main_data['season_name'] = main_data['season'].map(season_labels)
    
    # Weather mapping
    weather_labels = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    main_data['weather_condition'] = main_data['weathersit'].map(weather_labels)
    
    # Year mapping
    year_labels = {0: '2011', 1: '2012'}
    main_data['year_label'] = main_data['yr'].map(year_labels)
    
    # Workingday mapping
    workingday_labels = {0: 'Non-Working Day', 1: 'Working Day'}
    main_data['workingday_label'] = main_data['workingday'].map(workingday_labels)
    
    # Weekday mapping
    weekday_labels = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                      4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    main_data['weekday_name'] = main_data['weekday'].map(weekday_labels)
    
    # Month mapping
    month_labels = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    main_data['month_name'] = main_data['mnth'].map(month_labels)
    
    # Denormalize weather metrics to their actual values
    main_data['temp_actual'] = main_data['temp'] * 41  # Celsius
    main_data['atemp_actual'] = main_data['atemp'] * 50  # Celsius
    main_data['hum_actual'] = main_data['hum'] * 100  # Percentage
    main_data['windspeed_actual'] = main_data['windspeed'] * 67  # km/h
    
    # Create temperature bins
    main_data['temp_category'] = pd.cut(
        main_data['temp_actual'], 
        bins=[-20, 0, 10, 20, 30, 50],
        labels=['Very Cold (< 0°C)', 'Cold (0-10°C)', 'Mild (10-20°C)', 'Warm (20-30°C)', 'Hot (> 30°C)']
    )
    
    # Create humidity bins
    main_data['hum_category'] = pd.cut(
        main_data['hum_actual'], 
        bins=[0, 30, 60, 80, 100],
        labels=['Low (< 30%)', 'Medium (30-60%)', 'High (60-80%)', 'Very High (> 80%)']
    )
    
    # Create windspeed bins
    main_data['windspeed_category'] = pd.cut(
        main_data['windspeed_actual'], 
        bins=[0, 10, 20, 30, 70],
        labels=['Low (< 10 km/h)', 'Medium (10-20 km/h)', 'High (20-30 km/h)', 'Very High (> 30 km/h)']
    )
    
    # Create time categories
    def categorize_time(hour):
        if 6 <= hour <= 9:
            return 'Morning Peak'
        elif 10 <= hour <= 15:
            return 'Day Time'
        elif 16 <= hour <= 19:
            return 'Evening Peak'
        else:
            return 'Night Time'
    
    main_data['time_category'] = main_data['hr'].apply(categorize_time)
    
    # Buat indeks kenyamanan
    main_data['comfort_index'] = (
        (main_data['temp'] * 0.5) +  
        ((1 - main_data['hum']) * 0.3) +  
        ((1 - main_data['windspeed']) * 0.2)  
    )
    
    # Categorize comfort index
    main_data['comfort_category'] = pd.cut(
        main_data['comfort_index'], 
        bins=[0, 0.3, 0.5, 0.7, 1],
        labels=['Uncomfortable', 'Moderately Comfortable', 'Comfortable', 'Very Comfortable']
    )
    
    # User type categorization
    main_data['user_type'] = np.where(main_data['casual'] > main_data['registered'], 'Casual Dominant', 'Registered Dominant')

    return main_data

main_data = load_data()

# Date range filter
min_date = main_data['dteday'].min().date()
max_date = main_data['dteday'].max().date()

with st.sidebar.expander("Date Range", expanded=True):
    date_range = st.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Season filter
with st.sidebar.expander("Season", expanded=True):
    season_options = ['All Seasons'] + sorted(main_data['season_name'].unique().tolist())
    selected_season = st.selectbox("Select Season", season_options)

# Weather filter
with st.sidebar.expander("Weather Condition", expanded=True):
    weather_options = ['All Weather Conditions'] + sorted(main_data['weather_condition'].unique().tolist())
    selected_weather = st.selectbox("Select Weather Condition", weather_options)

# Day type filter
with st.sidebar.expander("Day Type", expanded=True):
    day_options = ['All Days', 'Working Day', 'Non-Working Day', 'Weekday', 'Weekend', 'Holiday']
    selected_day_type = st.selectbox("Select Day Type", day_options)

# Temperature range filter
with st.sidebar.expander("Temperature Range", expanded=True):
    temp_options = ['All Temperature Ranges'] + sorted(main_data['temp_category'].unique().tolist())
    selected_temp = st.selectbox("Select Temperature Range", temp_options)
    
    # Additional slider for more granular temperature filtering
    temp_range = st.slider(
        "Temperature Range (°C)",
        float(main_data['temp_actual'].min()),
        float(main_data['temp_actual'].max()),
        (float(main_data['temp_actual'].min()), float(main_data['temp_actual'].max()))
    )

# User type filter
with st.sidebar.expander("User Type", expanded=True):
    user_options = ['All Users', 'Casual Dominant', 'Registered Dominant', 'Balanced']
    selected_user_type = st.selectbox("Select User Type", user_options)

# Reset filters button
if st.sidebar.button("Reset All Filters"):
    selected_season = 'All Seasons'
    selected_weather = 'All Weather Conditions'
    selected_day_type = 'All Days'
    selected_temp = 'All Temperature Ranges'
    date_range = [min_date, max_date]
    selected_user_type = 'All Users'
    temp_range = (float(main_data['temp_actual'].min()), float(main_data['temp_actual'].max()))

# Apply filters
filtered_data = main_data.copy()

# Date filter
if len(date_range) == 2:
    filtered_data = filtered_data[(filtered_data['dteday'].dt.date >= date_range[0]) & 
                                 (filtered_data['dteday'].dt.date <= date_range[1])]

# Season filter
if selected_season != 'All Seasons':
    filtered_data = filtered_data[filtered_data['season_name'] == selected_season]

# Weather filter
if selected_weather != 'All Weather Conditions':
    filtered_data = filtered_data[filtered_data['weather_condition'] == selected_weather]

# Day type filter
if selected_day_type == 'Working Day':
    filtered_data = filtered_data[filtered_data['workingday'] == 1]
elif selected_day_type == 'Non-Working Day':
    filtered_data = filtered_data[filtered_data['workingday'] == 0]
elif selected_day_type == 'Weekday':
    filtered_data = filtered_data[(filtered_data['weekday'] >= 1) & (filtered_data['weekday'] <= 5)]
elif selected_day_type == 'Weekend':
    filtered_data = filtered_data[(filtered_data['weekday'] == 0) | (filtered_data['weekday'] == 6)]
elif selected_day_type == 'Holiday':
    filtered_data = filtered_data[filtered_data['holiday'] == 1]

# Temperature filter
if selected_temp != 'All Temperature Ranges':
    filtered_data = filtered_data[filtered_data['temp_category'] == selected_temp]

# Additional temperature range filter
filtered_data = filtered_data[(filtered_data['temp_actual'] >= temp_range[0]) & 
                             (filtered_data['temp_actual'] <= temp_range[1])]

# User type filter
if selected_user_type == 'Casual Dominant':
    filtered_data = filtered_data[filtered_data['casual'] > filtered_data['registered']]
elif selected_user_type == 'Registered Dominant':
    filtered_data = filtered_data[filtered_data['registered'] > filtered_data['casual']]
elif selected_user_type == 'Balanced':
    ratio_threshold = 0.1  # 10% difference
    filtered_data = filtered_data[abs(filtered_data['casual'] - filtered_data['registered']) / filtered_data['cnt'] < ratio_threshold]

# Display currently applied filters
st.sidebar.markdown("---")
st.sidebar.subheader("Active Filters")
active_filters = []

if len(date_range) == 2 and (date_range[0] != min_date or date_range[1] != max_date):
    active_filters.append(f"📅 Date: {date_range[0]} to {date_range[1]}")
if selected_season != 'All Seasons':
    active_filters.append(f"🍂 Season: {selected_season}")
if selected_weather != 'All Weather Conditions':
    active_filters.append(f"☁️ Weather: {selected_weather}")
if selected_day_type != 'All Days':
    active_filters.append(f"📆 Day Type: {selected_day_type}")
if selected_temp != 'All Temperature Ranges':
    active_filters.append(f"🌡️ Temperature: {selected_temp}")
if temp_range != (float(main_data['temp_actual'].min()), float(main_data['temp_actual'].max())):
    active_filters.append(f"🌡️ Temp Range: {temp_range[0]:.1f}°C to {temp_range[1]:.1f}°C")
if selected_user_type != 'All Users':
    active_filters.append(f"👥 User Type: {selected_user_type}")

if active_filters:
    for filter_text in active_filters:
        st.sidebar.markdown(f"- {filter_text}")
else:
    st.sidebar.markdown("No filters applied")

# Main dashboard title
st.title("🚲 Bike Rental Analysis Dashboard")
st.markdown("Exploring patterns in bike rentals based on weather conditions, seasons, and time factors.")

# Display filter summary at the top
if active_filters:
    filter_summary = " | ".join(active_filters)
    st.info(f"**Applied Filters:** {filter_summary}")

# Key metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rentals", f"{filtered_data['cnt'].sum():,}")
    
with col2:
    st.metric("Average Rentals per Hour", f"{filtered_data['cnt'].mean():.1f}")
    
with col3:
    casual_pct = filtered_data['casual'].sum()/filtered_data['cnt'].sum()*100
    st.metric("Casual Users", f"{filtered_data['casual'].sum():,} ({casual_pct:.1f}%)")
    
with col4:
    registered_pct = filtered_data['registered'].sum()/filtered_data['cnt'].sum()*100
    st.metric("Registered Users", f"{filtered_data['registered'].sum():,} ({registered_pct:.1f}%)")

# Add comfort index metric
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_temp = filtered_data['temp_actual'].mean()
    st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
with col2:
    avg_hum = filtered_data['hum_actual'].mean()
    st.metric("Avg Humidity", f"{avg_hum:.1f}%")
    
with col3:
    avg_wind = filtered_data['windspeed_actual'].mean()
    st.metric("Avg Wind Speed", f"{avg_wind:.1f} km/h")
    
with col4:
    avg_comfort = filtered_data['comfort_index'].mean()
    st.metric("Avg Comfort Index", f"{avg_comfort:.2f}/1.0")

# Weather Impact Analysis
st.header("1. Weather Impact on Bike Rentals")

tab1, tab2, tab3 = st.tabs(["Weather Conditions", "Temperature Effect", "Comfort Analysis"])

with tab1:
    # Distribution by weather condition
    st.subheader("Rentals by Weather Condition")
    
    weather_agg = filtered_data.groupby('weather_condition')['cnt'].mean().reset_index()
    fig = px.bar(weather_agg, x='weather_condition', y='cnt',
                color='weather_condition',
                labels={'cnt': 'Average Hourly Rentals', 'weather_condition': 'Weather Condition'},
                title='Average Rentals by Weather Condition')
    fig.update_layout(xaxis_title='Weather Condition', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Weather condition pattern throughout the day
    st.subheader("Hourly Rental Pattern by Weather Condition")
    
    hourly_weather = filtered_data.groupby(['weather_condition', 'hr'])['cnt'].mean().reset_index()
    fig = px.line(hourly_weather, x='hr', y='cnt', color='weather_condition',
                 labels={'hr': 'Hour of Day', 'cnt': 'Average Rentals', 'weather_condition': 'Weather Condition'},
                 title='Hourly Rental Pattern by Weather Condition')
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 24))),
                     xaxis_title='Hour of Day', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Temperature effect
    st.subheader("Temperature Effect on Rentals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_data, x='temp_actual', y='cnt',
                        color='season_name', size='cnt',
                        labels={'temp_actual': 'Temperature (°C)', 'cnt': 'Hourly Rentals', 'season_name': 'Season'},
                        title='Temperature vs. Rentals')
        fig.update_layout(xaxis_title='Temperature (°C)', yaxis_title='Hourly Rentals')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        temp_cat_agg = filtered_data.groupby('temp_category')['cnt'].mean().reset_index()
        fig = px.bar(temp_cat_agg, x='temp_category', y='cnt',
                    color='temp_category',
                    labels={'temp_category': 'Temperature Range', 'cnt': 'Average Rentals'},
                    title='Average Rentals by Temperature Category')
        fig.update_layout(xaxis_title='Temperature Category', yaxis_title='Average Rentals')
        st.plotly_chart(fig, use_container_width=True)
    
    # Humidity and wind effect
    st.subheader("Humidity & Wind Speed Effects")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_data, x='hum_actual', y='cnt',
                        color='season_name',
                        labels={'hum_actual': 'Humidity (%)', 'cnt': 'Hourly Rentals'},
                        title='Humidity vs. Rentals')
        fig.update_layout(xaxis_title='Humidity (%)', yaxis_title='Hourly Rentals')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(filtered_data, x='windspeed_actual', y='cnt',
                        color='season_name',
                        labels={'windspeed_actual': 'Wind Speed (km/h)', 'cnt': 'Hourly Rentals'},
                        title='Wind Speed vs. Rentals')
        fig.update_layout(xaxis_title='Wind Speed (km/h)', yaxis_title='Hourly Rentals')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Comfort index analysis
    st.subheader("Comfort Index Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_data, x='comfort_index', y='cnt',
                        color='season_name',
                        labels={'comfort_index': 'Comfort Index', 'cnt': 'Hourly Rentals', 'season_name': 'Season'},
                        title='Comfort Index vs. Rentals')
        fig.update_layout(xaxis_title='Comfort Index (0-1)', yaxis_title='Hourly Rentals')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        comfort_cat_agg = filtered_data.groupby('comfort_category')['cnt'].mean().reset_index()
        fig = px.bar(comfort_cat_agg, x='comfort_category', y='cnt',
                    color='comfort_category',
                    labels={'comfort_category': 'Comfort Category', 'cnt': 'Average Rentals'},
                    title='Average Rentals by Comfort Category')
        fig.update_layout(xaxis_title='Comfort Category', yaxis_title='Average Rentals')
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined weather metrics heatmap
    st.subheader("Combined Weather Metrics Impact")
    
    # Create temp-humidity pivot table
    temp_hum_pivot = pd.pivot_table(filtered_data, 
                                   values='cnt', 
                                   index='temp_category', 
                                   columns='hum_category', 
                                   aggfunc='mean')
    
    fig = px.imshow(temp_hum_pivot,
                   labels=dict(x="Humidity Category", y="Temperature Category", color="Average Rentals"),
                   x=temp_hum_pivot.columns,
                   y=temp_hum_pivot.index,
                   color_continuous_scale='YlGnBu',
                   title='Heatmap: Temperature vs Humidity Impact on Rentals')
    st.plotly_chart(fig, use_container_width=True)

# Seasonal and Temporal Patterns
st.header("2. Seasonal and Temporal Patterns")

tab1, tab2, tab3 = st.tabs(["Seasonal Analysis", "Daily Patterns", "Time Category Analysis"])

with tab1:
    # Seasonal distribution
    st.subheader("Rentals by Season")
    
    season_agg = filtered_data.groupby('season_name')['cnt'].mean().reset_index()
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_agg['season_name'] = pd.Categorical(season_agg['season_name'], categories=season_order, ordered=True)
    season_agg = season_agg.sort_values('season_name')
    
    fig = px.bar(season_agg, x='season_name', y='cnt',
                color='season_name',
                labels={'cnt': 'Average Hourly Rentals', 'season_name': 'Season'},
                title='Average Rentals by Season')
    fig.update_layout(xaxis_title='Season', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trend
    st.subheader("Monthly Rental Pattern")
    
    monthly_agg = filtered_data.groupby('month_name')['cnt'].mean().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly_agg['month_name'] = pd.Categorical(monthly_agg['month_name'], categories=month_order, ordered=True)
    monthly_agg = monthly_agg.sort_values('month_name')
    
    fig = px.line(monthly_agg, x='month_name', y='cnt', markers=True,
                 labels={'cnt': 'Average Hourly Rentals', 'month_name': 'Month'},
                 title='Monthly Rental Pattern')
    fig.update_layout(xaxis_title='Month', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Season-Weather interaction
    st.subheader("Season and Weather Interaction")
    
    season_weather = filtered_data.groupby(['season_name', 'weather_condition'])['cnt'].mean().reset_index()
    season_weather_pivot = pd.pivot_table(season_weather, values='cnt', 
                                         index='season_name', columns='weather_condition')
    
    fig = px.imshow(season_weather_pivot,
                   labels=dict(x="Weather Condition", y="Season", color="Average Rentals"),
                   x=season_weather_pivot.columns,
                   y=season_order,
                   color_continuous_scale='YlGnBu',
                   title='Heatmap: Season vs Weather Condition')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Weekday patterns
    st.subheader("Rentals by Day of Week")
    
    weekday_agg = filtered_data.groupby('weekday_name')['cnt'].mean().reset_index()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_agg['weekday_name'] = pd.Categorical(weekday_agg['weekday_name'], categories=weekday_order, ordered=True)
    weekday_agg = weekday_agg.sort_values('weekday_name')
    
    fig = px.bar(weekday_agg, x='weekday_name', y='cnt',
                color='weekday_name',
                labels={'cnt': 'Average Hourly Rentals', 'weekday_name': 'Day of Week'},
                title='Average Rentals by Day of Week')
    fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly pattern by day type
    st.subheader("Hourly Pattern by Day Type")
    
    hourly_day_type = filtered_data.groupby(['workingday_label', 'hr'])['cnt'].mean().reset_index()
    
    fig = px.line(hourly_day_type, x='hr', y='cnt', color='workingday_label', markers=True,
                 labels={'hr': 'Hour of Day', 'cnt': 'Average Rentals', 'workingday_label': 'Day Type'},
                 title='Hourly Rental Pattern by Day Type')
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 24))),
                     xaxis_title='Hour of Day', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap: Hour vs Day of Week
    st.subheader("Hourly Pattern by Day of Week")
    
    hour_weekday = filtered_data.groupby(['hr', 'weekday_name'])['cnt'].mean().reset_index()
    hour_weekday_pivot = pd.pivot_table(hour_weekday, values='cnt', 
                                      index='hr', columns='weekday_name')
    hour_weekday_pivot = hour_weekday_pivot[weekday_order]
    
    fig = px.imshow(hour_weekday_pivot,
                   labels=dict(x="Day of Week", y="Hour of Day", color="Average Rentals"),
                   x=weekday_order,
                   y=list(range(0, 24)),
                   color_continuous_scale='YlGnBu',
                   title='Heatmap: Hour of Day vs Day of Week')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Time category analysis
    st.subheader("Rentals by Time Category")
    
    time_cat_agg = filtered_data.groupby('time_category')['cnt'].mean().reset_index()
    time_cat_order = ['Morning Peak', 'Day Time', 'Evening Peak', 'Night Time']
    time_cat_agg['time_category'] = pd.Categorical(time_cat_agg['time_category'], categories=time_cat_order, ordered=True)
    time_cat_agg = time_cat_agg.sort_values('time_category')
    
    fig = px.bar(time_cat_agg, x='time_category', y='cnt',
                color='time_category',
                labels={'cnt': 'Average Hourly Rentals', 'time_category': 'Time Category'},
                title='Average Rentals by Time Category')
    fig.update_layout(xaxis_title='Time Category', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap: Season vs Time Category
    st.subheader("Season and Time Category Interaction")
    
    season_time = filtered_data.groupby(['season_name', 'time_category'])['cnt'].mean().reset_index()
    season_time_pivot = pd.pivot_table(season_time, values='cnt', 
                                      index='season_name', columns='time_category')
    
    # Ensure proper ordering
    if not season_time_pivot.empty:
        if all(season in season_time_pivot.index for season in season_order):
            season_time_pivot = season_time_pivot.reindex(season_order)
        if all(cat in season_time_pivot.columns for cat in time_cat_order):
            season_time_pivot = season_time_pivot[time_cat_order]
        
        fig = px.imshow(season_time_pivot,
                       labels=dict(x="Time Category", y="Season", color="Average Rentals"),
                       x=season_time_pivot.columns,
                       y=season_time_pivot.index,
                       color_continuous_scale='YlGnBu',
                       title='Heatmap: Season vs Time Category')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time category and weather interaction
    st.subheader("Time Category and Weather Interaction")
    
    time_weather = filtered_data.groupby(['time_category', 'weather_condition'])['cnt'].mean().reset_index()
    time_weather_pivot = pd.pivot_table(time_weather, values='cnt', 
                                       index='time_category', columns='weather_condition')
    
    if not time_weather_pivot.empty:
        fig = px.imshow(time_weather_pivot,
                       labels=dict(x="Weather Condition", y="Time Category", color="Average Rentals"),
                       x=time_weather_pivot.columns,
                       y=time_cat_order,
                       color_continuous_scale='YlGnBu',
                       title='Heatmap: Time Category vs Weather Condition')
        st.plotly_chart(fig, use_container_width=True)

# User Type Analysis
st.header("3. User Type Analysis")

tab1, tab2 = st.tabs(["Casual vs Registered", "User Patterns"])

with tab1:
    # Casual vs registered users
    st.subheader("Casual vs Registered Users")
    
    # Overall proportions
    user_props = pd.DataFrame({
        'User Type': ['Casual', 'Registered'],
        'Count': [filtered_data['casual'].sum(), filtered_data['registered'].sum()]
    })
    
    fig = px.pie(user_props, values='Count', names='User Type',
                title='Proportion of Casual vs Registered Users',
                color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
    
    # By season
    season_user = filtered_data.groupby('season_name')[['casual', 'registered']].mean().reset_index()
    season_user['season_name'] = pd.Categorical(season_user['season_name'], categories=season_order, ordered=True)
    season_user = season_user.sort_values('season_name')
    season_user_melted = pd.melt(season_user, id_vars='season_name', 
                                value_vars=['casual', 'registered'],
                                var_name='User Type', value_name='Average Rentals')
    
    fig = px.bar(season_user_melted, x='season_name', y='Average Rentals', color='User Type',
                barmode='group',
                labels={'season_name': 'Season'},
                title='Casual vs Registered Users by Season')
    fig.update_layout(xaxis_title='Season', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # By weather condition
    weather_user = filtered_data.groupby('weather_condition')[['casual', 'registered']].mean().reset_index()
    weather_user_melted = pd.melt(weather_user, id_vars='weather_condition', 
                                 value_vars=['casual', 'registered'],
                                 var_name='User Type', value_name='Average Rentals')
    
    fig = px.bar(weather_user_melted, x='weather_condition', y='Average Rentals', color='User Type',
                barmode='group',
                labels={'weather_condition': 'Weather Condition'},
                title='Casual vs Registered Users by Weather Condition')
    fig.update_layout(xaxis_title='Weather Condition', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # User patterns by day and hour
    st.subheader("User Patterns by Day and Hour")
    
    # By day of week
    weekday_user = filtered_data.groupby('weekday_name')[['casual', 'registered']].mean().reset_index()
    weekday_user['weekday_name'] = pd.Categorical(weekday_user['weekday_name'], categories=weekday_order, ordered=True)
    weekday_user = weekday_user.sort_values('weekday_name')
    weekday_user_melted = pd.melt(weekday_user, id_vars='weekday_name', 
                                 value_vars=['casual', 'registered'],
                                 var_name='User Type', value_name='Average Rentals')
    
    fig = px.bar(weekday_user_melted, x='weekday_name', y='Average Rentals', color='User Type',
                barmode='group',
                labels={'weekday_name': 'Day of Week'},
                title='Casual vs Registered Users by Day of Week')
    fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # By hour
    hourly_user = filtered_data.groupby('hr')[['casual', 'registered']].mean().reset_index()
    hourly_user_melted = pd.melt(hourly_user, id_vars='hr', 
                               value_vars=['casual', 'registered'],
                               var_name='User Type', value_name='Average Rentals')
    
    fig = px.line(hourly_user_melted, x='hr', y='Average Rentals', color='User Type', markers=True,
                 labels={'hr': 'Hour of Day'},
                 title='Hourly Pattern: Casual vs Registered Users')
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 24))),
                     xaxis_title='Hour of Day', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # By temperature category
    temp_user = filtered_data.groupby('temp_category')[['casual', 'registered']].mean().reset_index()
    temp_user_melted = pd.melt(temp_user, id_vars='temp_category', 
                              value_vars=['casual', 'registered'],
                              var_name='User Type', value_name='Average Rentals')
    
    fig = px.bar(temp_user_melted, x='temp_category', y='Average Rentals', color='User Type',
                barmode='group',
                labels={'temp_category': 'Temperature Category'},
                title='Casual vs Registered Users by Temperature Category')
    fig.update_layout(xaxis_title='Temperature Category', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # User ratio analysis
    st.subheader("User Ratio Analysis")
    
    # Calculate casual to registered ratio
    filtered_data['casual_ratio'] = filtered_data['casual'] / filtered_data['cnt']
    filtered_data['registered_ratio'] = filtered_data['registered'] / filtered_data['cnt']
    
    # Ratio by season and day of week
    ratio_pivot = pd.pivot_table(filtered_data, 
                                values='casual_ratio', 
                                index='season_name', 
                                columns='weekday_name', 
                                aggfunc='mean')
    
    # Ensure proper ordering
    if not ratio_pivot.empty:
        if all(season in ratio_pivot.index for season in season_order):
            ratio_pivot = ratio_pivot.reindex(season_order)
        if all(day in ratio_pivot.columns for day in weekday_order):
            ratio_pivot = ratio_pivot[weekday_order]
        
        fig = px.imshow(ratio_pivot,
                       labels=dict(x="Day of Week", y="Season", color="Casual User Ratio"),
                       x=ratio_pivot.columns,
                       y=ratio_pivot.index,
                       color_continuous_scale='RdBu_r',
                       title='Heatmap: Casual User Ratio by Season and Day of Week')
        st.plotly_chart(fig, use_container_width=True)

# Comfort and Environmental Analysis
st.header("4. Comfort and Environmental Analysis")

tab1, tab2 = st.tabs(["Comfort Index Analysis", "Environmental Factors"])

with tab1:
    # Comfort index analysis
    st.subheader("Comfort Index Impact on Rentals")
    
    # Comfort index by season
    comfort_season = filtered_data.groupby('season_name')['comfort_index'].mean().reset_index()
    comfort_season['season_name'] = pd.Categorical(comfort_season['season_name'], categories=season_order, ordered=True)
    comfort_season = comfort_season.sort_values('season_name')
    
    fig = px.bar(comfort_season, x='season_name', y='comfort_index',
                color='season_name',
                labels={'comfort_index': 'Average Comfort Index', 'season_name': 'Season'},
                title='Average Comfort Index by Season')
    fig.update_layout(xaxis_title='Season', yaxis_title='Comfort Index (0-1)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Comfort category distribution
    comfort_dist = filtered_data['comfort_category'].value_counts().reset_index()
    comfort_dist.columns = ['Comfort Category', 'Count']
    
    fig = px.pie(comfort_dist, values='Count', names='Comfort Category',
                title='Distribution of Comfort Categories',
                color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)
    
    # Comfort vs rentals scatter
    fig = px.scatter(filtered_data, x='comfort_index', y='cnt',
                    color='season_name', size='cnt', opacity=0.7,
                    labels={'comfort_index': 'Comfort Index', 'cnt': 'Hourly Rentals', 'season_name': 'Season'},
                    title='Relationship Between Comfort Index and Rentals')
    fig.update_layout(xaxis_title='Comfort Index (0-1)', yaxis_title='Hourly Rentals')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Environmental factors analysis
    st.subheader("Environmental Factors Impact")
    
    # Temperature distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(filtered_data, x='temp_actual',
                          nbins=20,
                          labels={'temp_actual': 'Temperature (°C)'},
                          title='Temperature Distribution')
        fig.update_layout(xaxis_title='Temperature (°C)', yaxis_title='Frequency')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(filtered_data, x='hum_actual',
                          nbins=20,
                          labels={'hum_actual': 'Humidity (%)'},
                          title='Humidity Distribution')
        fig.update_layout(xaxis_title='Humidity (%)', yaxis_title='Frequency')
        st.plotly_chart(fig, use_container_width=True)
    
    # Wind speed analysis
    st.subheader("Wind Speed Analysis")
    
    wind_cat_agg = filtered_data.groupby('windspeed_category')['cnt'].mean().reset_index()
    
    fig = px.bar(wind_cat_agg, x='windspeed_category', y='cnt',
                color='windspeed_category',
                labels={'windspeed_category': 'Wind Speed Category', 'cnt': 'Average Rentals'},
                title='Average Rentals by Wind Speed Category')
    fig.update_layout(xaxis_title='Wind Speed Category', yaxis_title='Average Rentals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Combined environmental factors
    st.subheader("Combined Environmental Factors")
    
    # Create a bubble chart
    fig = px.scatter(filtered_data, x='temp_actual', y='hum_actual',
                    size='cnt', color='season_name',
                    hover_name='dteday', hover_data=['windspeed_actual', 'cnt'],
                    labels={'temp_actual': 'Temperature (°C)', 
                           'hum_actual': 'Humidity (%)', 
                           'cnt': 'Rentals',
                           'season_name': 'Season',
                           'windspeed_actual': 'Wind Speed (km/h)'},
                    title='Combined Environmental Factors Impact on Rentals')
    fig.update_layout(xaxis_title='Temperature (°C)', yaxis_title='Humidity (%)')
    st.plotly_chart(fig, use_container_width=True)

# Interactive Exploration
st.header("5. Interactive Exploration")

tab1, tab2 = st.tabs(["3D Visualization", "Correlation Analysis"])

with tab1:
    # 3D scatter plot
    st.subheader("3D Exploration: Temperature, Humidity, and Rentals")
    
    fig = px.scatter_3d(filtered_data, x='temp_actual', y='hum_actual', z='cnt',
                       color='season_name',
                       size='cnt',
                       opacity=0.7,
                       labels={'temp_actual': 'Temperature (°C)', 
                              'hum_actual': 'Humidity (%)', 
                              'cnt': 'Rentals', 
                              'season_name': 'Season'})
    fig.update_layout(scene=dict(xaxis_title='Temperature (°C)',
                                yaxis_title='Humidity (%)',
                                zaxis_title='Hourly Rentals'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced 3D visualization
    st.subheader("Advanced 3D Visualization")
    
    viz_options = ['Temperature-Humidity-Rentals', 'Temperature-Wind-Rentals', 'Humidity-Wind-Rentals', 'Hour-Temperature-Rentals']
    selected_viz = st.selectbox("Select 3D Visualization", viz_options)
    
    if selected_viz == 'Temperature-Humidity-Rentals':
        fig = px.scatter_3d(filtered_data, x='temp_actual', y='hum_actual', z='cnt',
                           color='weather_condition',
                           size='cnt', opacity=0.7)
    elif selected_viz == 'Temperature-Wind-Rentals':
        fig = px.scatter_3d(filtered_data, x='temp_actual', y='windspeed_actual', z='cnt',
                           color='weather_condition',
                           size='cnt', opacity=0.7)
    elif selected_viz == 'Humidity-Wind-Rentals':
        fig = px.scatter_3d(filtered_data, x='hum_actual', y='windspeed_actual', z='cnt',
                           color='weather_condition',
                           size='cnt', opacity=0.7)
    else:  # Hour-Temperature-Rentals
        fig = px.scatter_3d(filtered_data, x='hr', y='temp_actual', z='cnt',
                           color='workingday_label',
                           size='cnt', opacity=0.7)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    
    numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt', 'casual', 'registered', 'comfort_index']
    corr_matrix = filtered_data[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                   labels=dict(color="Correlation Coefficient"),
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   color_continuous_scale='RdBu_r',
                   zmin=-1, zmax=1)
    fig.update_layout(title='Correlation Between Numerical Variables')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance for Rentals")
    
    # Sort correlation with cnt
    cnt_corr = corr_matrix['cnt'].drop('cnt').sort_values(ascending=False)
    cnt_corr = cnt_corr.reset_index()
    cnt_corr.columns = ['Feature', 'Correlation']
    
    fig = px.bar(cnt_corr, x='Feature', y='Correlation',
                color='Correlation',
                color_continuous_scale='RdBu_r',
                title='Feature Importance for Predicting Rentals')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Correlation with Rentals')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Temuan Utama

1. **Dampak Cuaca**:
   - Kondisi cuaca cerah menyebabkan peningkatan signifikan dalam penyewaan sepeda.
   - Suhu memiliki korelasi positif terkuat dengan jumlah penyewaan.
   - Kelembapan tinggi dan kecepatan angin berdampak negatif pada jumlah penyewaan.
   - Indeks kenyamanan menunjukkan korelasi kuat dengan pola penyewaan.

2. **Pola Musiman**:
   - Musim panas dan musim gugur menunjukkan aktivitas penyewaan tertinggi.
   - Musim dingin memiliki jumlah penyewaan terendah di semua kategori waktu.
   - Pola bulanan menunjukkan puncak penyewaan terjadi dari Juni hingga September.

3. **Pola Harian dan Jam Tertentu**:
   - Terdapat dua puncak penyewaan pada hari kerja: pagi (07.00-09.00) dan sore (17.00-19.00).
   - Pola akhir pekan menunjukkan satu puncak lebih luas pada siang hari.
   - Malam hari (23.00-05.00) secara konsisten menunjukkan aktivitas penyewaan terendah.

4. **Perilaku Pengguna**:
   - Pengguna terdaftar mendominasi penyewaan pada hari kerja, terutama pada jam-jam komuter.
   - Pengguna kasual lebih banyak berkontribusi pada akhir pekan dan hari libur.
   - Kondisi cuaca lebih berdampak pada pengguna kasual dibandingkan pengguna terdaftar.

### Rekomendasi

1. **Optimasi Operasional**:
   - Tingkatkan ketersediaan sepeda selama jam sibuk, terutama di musim panas dan gugur.
   - Pertimbangkan pengurangan operasional saat cuaca ekstrem.
   - Jadwalkan pemeliharaan sepeda pada malam hari ketika permintaan paling rendah.

2. **Strategi Pemasaran**:
   - Targetkan pengguna kasual dengan promosi akhir pekan.
   - Kembangkan kampanye khusus untuk meningkatkan penyewaan selama musim dingin.
   - Pertimbangkan harga dinamis berbasis cuaca.

3. **Peningkatan Pengalaman Pengguna**:
   - Pastikan ketersediaan sepeda yang cukup di pusat transportasi selama jam sibuk.
   - Integrasikan prakiraan cuaca dalam aplikasi penyewaan sepeda.
   - Kembangkan program loyalitas bagi pengguna rutin.
""")