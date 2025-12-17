import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Airline Delay Analyzer",
    page_icon="✈️",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    flights = pd.read_csv('data/raw/flights_2019.csv', low_memory=False)
    return flights

@st.cache_resource
def load_models():
    with open('src/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('src/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

# Cost calculator
def calculate_delay_cost(delay_minutes, passengers=150):
    fuel_cost = delay_minutes * 40
    crew_cost = delay_minutes * 25
    maintenance_cost = delay_minutes * 15
    
    if delay_minutes >= 180:
        passenger_compensation = passengers * 75
        nps_impact = -30
    elif delay_minutes >= 120:
        passenger_compensation = passengers * 25
        nps_impact = -20
    elif delay_minutes >= 60:
        passenger_compensation = passengers * 10
        nps_impact = -10
    else:
        passenger_compensation = 0
        nps_impact = -2 if delay_minutes > 0 else 0
    
    total_cost = fuel_cost + crew_cost + maintenance_cost + passenger_compensation
    
    return {
        "fuel_cost": fuel_cost,
        "crew_cost": crew_cost,
        "maintenance_cost": maintenance_cost,
        "passenger_compensation": passenger_compensation,
        "total_cost": total_cost,
        "nps_impact": nps_impact
    }

# Load data
flights = load_data()
model, scaler, features = load_models()

# Sidebar
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Delay Predictor", "Cost Simulator", "Route Analyzer"])

# ============================================
# Dashboard Page
# ============================================
if page == "Dashboard":
    st.title("✈️ Airline Delay Analytics Dashboard")
    st.markdown("---")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Flights", f"{len(flights):,}")
    with col2:
        delay_rate = flights['DEP_DEL15'].mean() * 100
        st.metric("Delay Rate", f"{delay_rate:.1f}%")
    with col3:
        total_delayed = flights['DEP_DEL15'].sum()
        st.metric("Delayed Flights", f"{int(total_delayed):,}")
    with col4:
        num_airports = flights['DEPARTING_AIRPORT'].nunique()
        st.metric("Airports", f"{num_airports}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delay Rate by Month")
        monthly = flights.groupby('MONTH')['DEP_DEL15'].mean() * 100
        fig = px.bar(x=monthly.index, y=monthly.values, 
                     labels={'x': 'Month', 'y': 'Delay Rate (%)'},
                     color=monthly.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Delay Rate by Day of Week")
        days = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
        daily = flights.groupby('DAY_OF_WEEK')['DEP_DEL15'].mean() * 100
        daily.index = daily.index.map(days)
        fig = px.bar(x=daily.index, y=daily.values,
                     labels={'x': 'Day', 'y': 'Delay Rate (%)'},
                     color=daily.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Airlines
    st.subheader("Airline Performance")
    carrier_delays = flights.groupby('CARRIER_NAME')['DEP_DEL15'].mean().sort_values() * 100
    fig = px.bar(x=carrier_delays.values, y=carrier_delays.index, orientation='h',
                 labels={'x': 'Delay Rate (%)', 'y': 'Airline'},
                 color=carrier_delays.values, color_continuous_scale='RdYlGn_r')
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# Delay Predictor Page (FIXED)
# ============================================
elif page == "Delay Predictor":
    st.title("🔮 Flight Delay Predictor")
    st.markdown("Predict if your flight will be delayed")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        month = st.selectbox("Month", range(1, 13), format_func=lambda x: 
                            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
        
        day = st.selectbox("Day of Week", range(1, 8), format_func=lambda x:
                          ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday'][x-1])
        
        airport = st.selectbox("Departure Airport", 
                               sorted(flights['DEPARTING_AIRPORT'].unique()))
    
    with col2:
        carrier = st.selectbox("Airline", 
                               sorted(flights['CARRIER_NAME'].unique()))
    
    if st.button("Predict Delay", type="primary"):
        # Try different filter combinations (less strict)
        sample = flights[
            (flights['CARRIER_NAME'] == carrier) &
            (flights['DEPARTING_AIRPORT'] == airport)
        ]
        
        # If no match, try just carrier
        if len(sample) == 0:
            sample = flights[flights['CARRIER_NAME'] == carrier]
            st.info(f"Using data from all {carrier} flights")
        
        # If still no match, try just airport
        if len(sample) == 0:
            sample = flights[flights['DEPARTING_AIRPORT'] == airport]
            st.info(f"Using data from all flights at {airport}")
        
        # Last resort - use all data
        if len(sample) == 0:
            sample = flights.sample(n=10000)
            st.info("Using general flight data for prediction")
        
        if len(sample) > 0:
            # Get numeric and categorical columns
            sample_features = sample[features].copy()
            numeric_cols = sample_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = sample_features.select_dtypes(include=['object']).columns.tolist()
            
            # Build input data
            input_dict = {}
            for col in numeric_cols:
                input_dict[col] = sample_features[col].median()
            for col in categorical_cols:
                if len(sample_features[col].mode()) > 0:
                    input_dict[col] = sample_features[col].mode().iloc[0]
                else:
                    input_dict[col] = sample_features[col].iloc[0]
            
            input_data = pd.DataFrame([input_dict])
            input_data['MONTH'] = month
            input_data['DAY_OF_WEEK'] = day
            
            # Encode categorical
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(flights[col].astype(str))
                input_data[col] = le.transform(input_data[col].astype(str))
            
            # Ensure correct column order
            input_data = input_data[features]
            
            # Predict
            input_scaled = scaler.transform(input_data)
            probability = model.predict_proba(input_scaled)[0][1] * 100
            
            # Get historical rate
            historical = sample['DEP_DEL15'].mean() * 100
            
            # Display result
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Delay Probability", f"{probability:.1f}%")
            with col2:
                if probability > 50:
                    risk = "🔴 HIGH"
                elif probability > 30:
                    risk = "🟡 MEDIUM"
                else:
                    risk = "🟢 LOW"
                st.metric("Risk Level", risk)
            with col3:
                st.metric("Historical Rate", f"{historical:.1f}%")
            
            # Recommendation
            if probability > 50:
                st.error("⚠️ High delay risk! Consider booking an earlier flight or different day.")
            elif probability > 30:
                st.warning("🔶 Moderate delay risk. Have a backup plan.")
            else:
                st.success("✅ Low delay risk. Your flight looks good!")
            
            # Show sample size
            st.caption(f"Based on {len(sample):,} historical flights")

# ============================================
# Cost Simulator Page
# ============================================
elif page == "Cost Simulator":
    st.title("💰 Delay Cost Simulator")
    st.markdown("Calculate the financial impact of flight delays")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delay_minutes = st.slider("Delay Duration (minutes)", 0, 300, 60)
        passengers = st.slider("Number of Passengers", 50, 400, 150)
    
    with col2:
        aircraft = st.selectbox("Aircraft Type", ["Narrow Body (737, A320)", "Wide Body (777, A350)"])
    
    # Calculate
    result = calculate_delay_cost(delay_minutes, passengers)
    
    st.markdown("---")
    st.subheader("Cost Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fuel Cost", f"${result['fuel_cost']:,}")
    with col2:
        st.metric("Crew Cost", f"${result['crew_cost']:,}")
    with col3:
        st.metric("Passenger Comp.", f"${result['passenger_compensation']:,}")
    with col4:
        st.metric("Total Cost", f"${result['total_cost']:,}")
    
    # Chart
    if delay_minutes > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Fuel', 'Crew', 'Maintenance', 'Passenger Comp.'],
            values=[result['fuel_cost'], result['crew_cost'], 
                    result['maintenance_cost'], result['passenger_compensation']],
            hole=0.4
        )])
        fig.update_layout(title="Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # NPS Impact
    st.metric("Customer Satisfaction Impact (NPS)", f"{result['nps_impact']} points")
    
    # Annual impact calculator
    st.markdown("---")
    st.subheader("Annual Impact Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        flights_per_day = st.number_input("Delayed Flights Per Day", min_value=1, value=50)
    with col2:
        avg_delay = st.number_input("Average Delay (minutes)", min_value=1, value=45)
    
    annual_cost = calculate_delay_cost(avg_delay, passengers)['total_cost'] * flights_per_day * 365
    st.metric("Estimated Annual Delay Cost", f"${annual_cost:,.0f}")

# ============================================
# Route Analyzer Page
# ============================================
elif page == "Route Analyzer":
    st.title("🗺️ Route Analyzer")
    st.markdown("Analyze delay patterns by airport")
    st.markdown("---")
    
    airport = st.selectbox("Select Airport", 
                           sorted(flights['DEPARTING_AIRPORT'].unique()))
    
    if st.button("Analyze", type="primary"):
        airport_data = flights[flights['DEPARTING_AIRPORT'] == airport]
        
        if len(airport_data) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Flights", f"{len(airport_data):,}")
            with col2:
                delay_rate = airport_data['DEP_DEL15'].mean() * 100
                st.metric("Delay Rate", f"{delay_rate:.1f}%")
            with col3:
                delayed = airport_data['DEP_DEL15'].sum()
                st.metric("Delayed Flights", f"{int(delayed):,}")
            
            # Estimated annual cost
            avg_delay_cost = calculate_delay_cost(45, 150)['total_cost']
            annual_cost = delayed * avg_delay_cost
            st.metric("Estimated Annual Delay Cost", f"${annual_cost:,.0f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Delays by Month")
                monthly = airport_data.groupby('MONTH')['DEP_DEL15'].mean() * 100
                fig = px.line(x=monthly.index, y=monthly.values,
                             labels={'x': 'Month', 'y': 'Delay Rate (%)'}, markers=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Best/Worst month
                best_month = monthly.idxmin()
                worst_month = monthly.idxmax()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                st.success(f"✅ Best Month: {month_names[best_month-1]} ({monthly[best_month]:.1f}%)")
                st.error(f"❌ Worst Month: {month_names[worst_month-1]} ({monthly[worst_month]:.1f}%)")
            
            with col2:
                st.subheader("Top Carriers at This Airport")
                carriers = airport_data.groupby('CARRIER_NAME').agg({
                    'DEP_DEL15': ['mean', 'count']
                }).reset_index()
                carriers.columns = ['Carrier', 'Delay Rate', 'Flights']
                carriers['Delay Rate'] = (carriers['Delay Rate'] * 100).round(1)
                carriers = carriers.sort_values('Flights', ascending=False).head(10)
                st.dataframe(carriers, use_container_width=True, hide_index=True)
                
                # Best carrier
                if len(carriers) > 0:
                    best_carrier = carriers.loc[carriers['Delay Rate'].idxmin(), 'Carrier']
                    best_rate = carriers['Delay Rate'].min()
                    st.success(f"✅ Best Carrier: {best_carrier} ({best_rate:.1f}% delays)")
            
            # Day of week analysis
            st.markdown("---")
            st.subheader("Delays by Day of Week")
            days = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                    5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
            daily = airport_data.groupby('DAY_OF_WEEK')['DEP_DEL15'].mean() * 100
            daily.index = daily.index.map(days)
            fig = px.bar(x=daily.index, y=daily.values,
                        labels={'x': 'Day', 'y': 'Delay Rate (%)'},
                        color=daily.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# Footer
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Built with Streamlit")
st.sidebar.markdown("Data: 2019 US Flight Delays")
st.sidebar.markdown("4.5M+ flights analyzed")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown("Ensemble Model: XGBoost + LightGBM + Random Forest")
st.sidebar.markdown("Accuracy: 82.67%")