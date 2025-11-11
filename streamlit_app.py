import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Configure the app
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #FF9900;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚚 Amazon Delivery Time Prediction</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h3 style='color: #666;'>Accurate Delivery Time Predictions Using Machine Learning</h3>
</div>
""", unsafe_allow_html=True)

# Load model and info
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_delivery_model.pkl')
        model_info = joblib.load('models/model_info.pkl')
        return model, model_info
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.info("""
        **Please run these steps first:**
        1. `python fixed_data_preparation.py`
        2. `python fixed_model_training.py`
        """)
        return None, None

model, model_info = load_model()

if model is not None:
    # Display model info
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.success(f"""
    **Best Model:** {model_info.get('best_model', 'N/A')}
    **RMSE:** {model_info.get('best_rmse', 'N/A'):.3f} hours
    **Features:** {len(model_info.get('features', []))}
    """)
    
    # Main input form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📍 Location Details")
        
        # Store coordinates (Amazon warehouse)
        store_lat = st.number_input("**Store Latitude**", value=28.6139, format="%.6f", key="store_lat")
        store_lng = st.number_input("**Store Longitude**", value=77.2090, format="%.6f", key="store_lng")
        
        # Delivery coordinates (Customer location)
        drop_lat = st.number_input("**Delivery Latitude**", value=28.4595, format="%.6f", key="drop_lat") 
        drop_lng = st.number_input("**Delivery Longitude**", value=77.0266, format="%.6f", key="drop_lng")
        
        # Calculate distance
        try:
            distance = geodesic((store_lat, store_lng), (drop_lat, drop_lng)).km
            st.metric("**📏 Calculated Distance**", f"{distance:.2f} km", delta=None)
        except Exception as e:
            distance = 15.0
            st.metric("**📏 Default Distance**", f"{distance:.2f} km")
        
        st.markdown("### 👤 Agent Information")
        agent_age = st.slider("**Agent Age**", 18, 60, 32)
        agent_rating = st.slider("**Agent Rating**", 1.0, 5.0, 4.3, 0.1)
    
    with col2:
        st.markdown("### 🌦️ Delivery Conditions")
        
        # Dynamic conditions
        weather = st.selectbox("**Weather Condition**", 
                              options=['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'])
        traffic = st.selectbox("**Traffic Condition**", 
                              options=['Low', 'Medium', 'High', 'Heavy', 'Jam'])
        vehicle = st.selectbox("**Vehicle Type**", 
                              options=['Bike', 'Scooter', 'Car', 'Van', 'Truck'])
        area = st.selectbox("**Area Type**", 
                           options=['Urban', 'Suburban', 'Metropolitan', 'Rural'])
        category = st.selectbox("**Product Category**", 
                               options=['Electronics', 'Clothing', 'Groceries', 'Books', 
                                       'Home & Kitchen', 'Sports', 'Beauty', 'Toys'])
        
        st.markdown("### ⏰ Order Timing")
        order_hour = st.slider("**Order Hour**", 0, 23, 14, 
                              help="Time when order was placed")
        order_dow = st.selectbox("**Order Day**", 
                               ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                'Friday', 'Saturday', 'Sunday'])
        processing_time = st.slider("**Processing Time (hours)**", 0.1, 5.0, 1.2, 0.1,
                                   help="Time from order to pickup")
    
    # Convert inputs to model features
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    order_dayofweek = day_mapping[order_dow]
    is_weekend = 1 if order_dow in ['Saturday', 'Sunday'] else 0
    
    # Encode categorical variables (simple mapping)
    weather_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Stormy': 3, 'Foggy': 4}
    traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Heavy': 3, 'Jam': 4}
    vehicle_mapping = {'Bike': 0, 'Scooter': 1, 'Car': 2, 'Van': 3, 'Truck': 4}
    area_mapping = {'Urban': 0, 'Suburban': 1, 'Metropolitan': 2, 'Rural': 3}
    category_mapping = {'Electronics': 0, 'Clothing': 1, 'Groceries': 2, 'Books': 3,
                       'Home & Kitchen': 4, 'Sports': 5, 'Beauty': 6, 'Toys': 7}
    
    # Create feature array
    input_features = np.array([[
        distance, agent_age, agent_rating, order_hour, order_dayofweek,
        weather_mapping[weather], traffic_mapping[traffic], 
        vehicle_mapping[vehicle], area_mapping[area], category_mapping[category],
        processing_time, is_weekend
    ]])
    
    # Prediction button
    st.markdown("---")
    if st.button("**🎯 PREDICT DELIVERY TIME**", type="primary", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(input_features)[0]
            
            # Ensure realistic prediction
            prediction = max(0.5, min(prediction, 48.0))
            
            # Convert to hours and minutes
            hours = int(prediction)
            minutes = int((prediction - hours) * 60)
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style='margin: 0; font-size: 2.5rem;'>📦 ESTIMATED DELIVERY TIME</h2>
                <h1 style='font-size: 4rem; margin: 1rem 0; color: #FFD700;'>{prediction:.1f} hours</h1>
                <p style='font-size: 1.2rem; margin: 0;'>Approximately {hours} hours {minutes} minutes</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Smart insights
            st.markdown("### 📊 DELIVERY INSIGHTS")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Distance impact (realistic calculation)
                base_time = distance * 0.08  # 5 minutes per km
                st.metric("**Distance Impact**", f"~{base_time:.1f}h", 
                         delta=f"{base_time/prediction*100:.0f}% of total")
            
            with col2:
                # Traffic impact
                traffic_impact = {'Low': 0.0, 'Medium': 0.2, 'High': 0.5, 'Heavy': 1.0, 'Jam': 2.0}
                impact = traffic_impact[traffic]
                st.metric("**Traffic Impact**", f"+{impact:.1f}h", 
                         delta="High delay" if impact > 0.5 else "Normal")
            
            with col3:
                # Weather impact
                weather_impact = {'Sunny': 0.0, 'Cloudy': 0.1, 'Rainy': 0.4, 'Stormy': 1.2, 'Foggy': 0.3}
                impact = weather_impact[weather]
                st.metric("**Weather Impact**", f"+{impact:.1f}h", 
                         delta="Severe" if impact > 0.5 else "Moderate")
            
            # Smart recommendations
            st.markdown("### 💡 SMART RECOMMENDATIONS")
            
            recommendations = []
            
            # Time-based recommendation
            if order_hour in [8, 9, 17, 18]:
                recommendations.append("**⏰ Avoid peak hours** (8-10 AM, 5-7 PM) for faster delivery")
            else:
                recommendations.append("**⏰ Good timing!** Current hour has normal traffic")
            
            # Vehicle recommendation
            if distance < 8 and area in ['Urban', 'Suburban']:
                recommendations.append("**🚲 Bike/Scooter recommended** for short urban distances")
            elif distance > 25:
                recommendations.append("**🚗 Car/Van recommended** for longer distances")
            
            # Area recommendation
            if area == 'Metropolitan':
                recommendations.append("**🏙️ Metropolitan area** - Expect moderate delays due to traffic")
            elif area == 'Rural':
                recommendations.append("**🌳 Rural area** - Delivery might take longer due to distance")
            
            # Weather recommendation
            if weather in ['Rainy', 'Stormy']:
                recommendations.append("**🌧️ Bad weather** - Delivery might be slower than usual")
            
            # Display recommendations
            for rec in recommendations:
                st.markdown(f"<div class='insight-box'>{rec}</div>", unsafe_allow_html=True)
            
            # Additional info
            st.markdown("### 🔍 PREDICTION DETAILS")
            st.info(f"""
            **Input Summary:**
            - **Distance:** {distance:.1f} km | **Vehicle:** {vehicle}
            - **Traffic:** {traffic} | **Weather:** {weather}
            - **Area:** {area} | **Product:** {category}
            - **Agent:** {agent_age}y, {agent_rating}⭐ rating
            - **Order Time:** {order_hour:02d}:00, {order_dow}
            """)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check if all required features are available.")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 🚀 How It Works")
        st.write("""
        This ML system predicts delivery times using:
        - **Real-time distance** calculations
        - **Traffic & weather** conditions  
        - **Agent performance** metrics
        - **Historical data** patterns
        - **Smart feature** engineering
        """)
        
        st.markdown("### 📈 Performance")
        st.write(f"""
        - **Accuracy:** ±{model_info.get('best_rmse', 0.5):.1f} hours
        - **Model:** {model_info.get('best_model', 'Optimized')}
        - **Training Data:** 40,000+ records
        """)
        
        st.markdown("### 🔧 Features Used")
        features = model_info.get('features', [])
        for feature in features[:6]:  # Show first 6 features
            st.write(f"• {feature}")

else:
    # Setup instructions
    st.warning("""
    ## ⚠️ SETUP REQUIRED
    
    **To use this prediction system:**
    
    1. **Prepare Data:**
       ```bash
       python fixed_data_preparation.py
       ```
    
    2. **Train Model:**
       ```bash  
       python fixed_model_training.py
       ```
    
    3. **Launch App:**
       ```bash
       streamlit run perfect_streamlit_app.py
       ```
    
    4. **View MLflow Dashboard:**
       ```bash
       mlflow ui
       ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using Streamlit & Scikit-learn | Amazon Delivery Time Prediction System"
    "</div>", 
    unsafe_allow_html=True
)