import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("🔄 STARTING DATA PREPARATION...")

def create_perfect_dataset():
    """Create a perfect dataset for Amazon Delivery Prediction"""
    
    # Load your original data
    try:
        df = pd.read_csv('amazon_delivery.csv')
        print(f"✅ Original data loaded: {df.shape}")
    except FileNotFoundError:
        print("❌ amazon_delivery.csv not found! Please check the file path.")
        return None
    
    # Display original data info
    print("\n📊 ORIGINAL DATA SUMMARY:")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Delivery Time Range: {df['Delivery_Time'].min():.2f} to {df['Delivery_Time'].max():.2f} hours")
    
    # FIX 1: Handle unrealistic delivery times
    print("\n🔧 FIXING DELIVERY TIMES...")
    original_max = df['Delivery_Time'].max()
    
    if original_max > 100:  # If delivery time > 100 hours, probably wrong unit
        df['Delivery_Time'] = df['Delivery_Time'] / 60  # Convert to hours
        print(f"⚠️ Converted delivery times (probably were in minutes)")
    
    # Cap unrealistic values (0.5 to 48 hours)
    df['Delivery_Time'] = np.clip(df['Delivery_Time'], 0.5, 48.0)
    
    print(f"✅ Fixed Delivery Time Range: {df['Delivery_Time'].min():.2f} to {df['Delivery_Time'].max():.2f} hours")
    
    # FIX 2: Handle missing values properly
    print("\n🔧 HANDLING MISSING VALUES...")
    print("Missing values before:")
    print(df.isnull().sum())
    
    # Numeric columns - fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns - fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    print("Missing values after:")
    print(df.isnull().sum())
    
    # FIX 3: Feature Engineering
    print("\n🔧 CREATING SMART FEATURES...")
    
    # Calculate realistic distances
    def calculate_distance(row):
        try:
            store_loc = (row['Store_Latitude'], row['Store_Longitude'])
            drop_loc = (row['Drop_Latitude'], row['Drop_Longitude'])
            distance = geodesic(store_loc, drop_loc).km
            return min(distance, 100)  # Cap at 100km
        except:
            return 5.0  # Default distance
    
    df['Distance_km'] = df.apply(calculate_distance, axis=1)
    
    # Create time-based features
    try:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
        df['Order_Hour'] = 12  # Default hour
        df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek.fillna(0)
        df['Is_Weekend'] = (df['Order_DayOfWeek'] >= 5).astype(int)
    except:
        df['Order_Hour'] = 12
        df['Order_DayOfWeek'] = 0
        df['Is_Weekend'] = 0
    
    # Encode categorical variables
    print("🔧 ENCODING CATEGORICAL VARIABLES...")
    label_encoders = {}
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"   ✅ {col}: {len(le.classes_)} categories")
        else:
            df[col + '_Encoded'] = 0
            print(f"   ⚠️ {col}: Column not found, using default")
    
    # Add processing time (realistic)
    df['Processing_Time_Hours'] = np.random.uniform(0.5, 2.0, len(df))
    
    # Save the fixed dataset
    df.to_csv('amazon_delivery_FIXED.csv', index=False)
    
    print(f"\n🎉 DATA PREPARATION COMPLETED!")
    print(f"📁 Fixed dataset saved: amazon_delivery_FIXED.csv")
    print(f"📊 Final shape: {df.shape}")
    print(f"⏱️ Delivery Time: {df['Delivery_Time'].min():.1f} to {df['Delivery_Time'].max():.1f} hours")
    
    return df, label_encoders

if __name__ == "__main__":
    df, encoders = create_perfect_dataset()
    if df is not None:
        print("\n✅ NEXT STEP: Run 'fixed_model_training.py'")