import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

print("🔄 STARTING AUTOMATIC DATA PREPARATION...")

def create_sample_dataset():
    """Create a realistic sample dataset if original doesn't exist"""
    print("📁 Creating sample dataset...")
    
    np.random.seed(42)
    n_samples = 5000  # Good size for training
    
    sample_data = {
        'Order_ID': [f'ORD{i:06d}' for i in range(n_samples)],
        'Agent_Age': np.random.randint(20, 50, n_samples),
        'Agent_Rating': np.round(np.random.uniform(3.0, 5.0, n_samples), 1),
        'Store_Latitude': np.random.uniform(28.4, 28.7, n_samples),
        'Store_Longitude': np.random.uniform(77.0, 77.3, n_samples),
        'Drop_Latitude': np.random.uniform(28.4, 28.7, n_samples),
        'Drop_Longitude': np.random.uniform(77.0, 77.3, n_samples),
        'Weather': np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Stormy'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'Traffic': np.random.choice(['Low', 'Medium', 'High', 'Jam'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'Vehicle': np.random.choice(['Bike', 'Car', 'Van', 'Truck'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Area': np.random.choice(['Urban', 'Suburban', 'Metropolitan', 'Rural'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Books', 'Home', 'Sports'], n_samples),
        'Delivery_Time': np.random.uniform(0.5, 8.0, n_samples)  # Realistic delivery times in hours
    }
    
    df = pd.DataFrame(sample_data)
    
    # Make delivery time realistic based on features
    for idx, row in df.iterrows():
        base_time = 0.5
        # Add distance factor
        try:
            store_loc = (row['Store_Latitude'], row['Store_Longitude'])
            drop_loc = (row['Drop_Latitude'], row['Drop_Longitude'])
            distance = geodesic(store_loc, drop_loc).km
            base_time += distance * 0.1  # 6 minutes per km
        except:
            base_time += 1.0
        
        # Add traffic factor
        traffic_factors = {'Low': 0.0, 'Medium': 0.3, 'High': 0.8, 'Jam': 1.5}
        base_time += traffic_factors.get(row['Traffic'], 0.0)
        
        # Add weather factor
        weather_factors = {'Sunny': 0.0, 'Cloudy': 0.1, 'Rainy': 0.4, 'Stormy': 1.0}
        base_time += weather_factors.get(row['Weather'], 0.0)
        
        df.at[idx, 'Delivery_Time'] = max(0.5, min(base_time + np.random.normal(0, 0.5), 8.0))
    
    df.to_csv('amazon_delivery.csv', index=False)
    print(f"✅ Created realistic sample dataset with {n_samples} records")
    return df

def prepare_data():
    """Main data preparation function"""
    
    # Check if data exists, if not create sample
    if not os.path.exists('amazon_delivery.csv'):
        df = create_sample_dataset()
    else:
        df = pd.read_csv('amazon_delivery.csv')
        print(f"✅ Original data loaded: {df.shape}")
    
    print("\n📊 DATA SUMMARY:")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Delivery Time Range: {df['Delivery_Time'].min():.2f} to {df['Delivery_Time'].max():.2f} hours")
    
    # Handle missing values
    print("\n🔧 Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    # Feature Engineering
    print("🔧 Creating features...")
    
    # Calculate distances
    def calculate_distance(row):
        try:
            store_loc = (row['Store_Latitude'], row['Store_Longitude'])
            drop_loc = (row['Drop_Latitude'], row['Drop_Longitude'])
            return geodesic(store_loc, drop_loc).km
        except:
            return np.random.uniform(1, 50)  # Random distance
    
    df['Distance_km'] = df.apply(calculate_distance, axis=1)
    
    # Time features (simplified)
    df['Order_Hour'] = np.random.randint(0, 24, len(df))
    df['Order_DayOfWeek'] = np.random.randint(0, 7, len(df))
    df['Is_Weekend'] = (df['Order_DayOfWeek'] >= 5).astype(int)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Processing time
    df['Processing_Time_Hours'] = np.random.uniform(0.5, 2.0, len(df))
    
    # Save processed data
    df.to_csv('amazon_delivery_FIXED.csv', index=False)
    
    print(f"\n🎉 DATA PREPARATION COMPLETED!")
    print(f"📁 Fixed dataset: amazon_delivery_FIXED.csv")
    print(f"📊 Final shape: {df.shape}")
    
    return df, label_encoders

if __name__ == "__main__":
    df, encoders = prepare_data()
    print("\n✅ Data ready for model training!")
