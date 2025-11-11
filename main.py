import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import streamlit as st
import joblib
from datetime import datetime
import sys

print("Python Version:", sys.version)
print("✅ All libraries imported successfully!")

class AmazonDeliveryPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_clean_data(self):
        """Step 1: Data Loading and Cleaning"""
        print("📊 Loading and cleaning data...")
        
        try:
            # Load dataset
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
        except FileNotFoundError:
            print(f"❌ Error: File '{self.data_path}' not found!")
            print("Please make sure 'amazon_delivery.csv' is in the same folder")
            return None
        
        print("\nMissing values before cleaning:")
        print(self.df.isnull().sum())
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        final_rows = len(self.df)
        print(f"Removed {initial_rows - final_rows} duplicate rows")
        
        print(f"\n✅ Data cleaning completed! Final shape: {self.df.shape}")
        return self.df
    
    def perform_feature_engineering(self):
        """Step 2: Feature Engineering"""
        print("⚙️ Performing feature engineering...")
        
        if self.df is None:
            print("❌ No data available. Please load data first.")
            return None
        
        # 1. Calculate distance using geopy
        def calculate_distance(row):
            try:
                store_loc = (row['Store_Latitude'], row['Store_Longitude'])
                drop_loc = (row['Drop_Latitude'], row['Drop_Longitude'])
                return geodesic(store_loc, drop_loc).km
            except Exception as e:
                print(f"Warning: Error calculating distance: {e}")
                return 5.0  # Default distance
        
        self.df['Distance_km'] = self.df.apply(calculate_distance, axis=1)
        
        # 2. Time-based features (if columns exist)
        if 'Order_Time' in self.df.columns and 'Order_Date' in self.df.columns:
            try:
                self.df['Order_Hour'] = pd.to_datetime(self.df['Order_Time'].astype(str)).dt.hour
                self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'])
                self.df['Order_DayOfWeek'] = self.df['Order_Date'].dt.dayofweek
                self.df['Is_Weekend'] = self.df['Order_DayOfWeek'].isin([5, 6]).astype(int)
            except Exception as e:
                print(f"Warning: Error processing time features: {e}")
                self.df['Order_Hour'] = 12
                self.df['Order_DayOfWeek'] = 0
                self.df['Is_Weekend'] = 0
        else:
            print("⚠️ Time columns not found, using default values")
            self.df['Order_Hour'] = 12
            self.df['Order_DayOfWeek'] = 0
            self.df['Is_Weekend'] = 0
        
        # 3. Encode categorical variables
        categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
        for col in categorical_cols:
            if col in self.df.columns:
                try:
                    le = LabelEncoder()
                    self.df[col + '_Encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"✅ Encoded {col}: {len(le.classes_)} categories")
                except Exception as e:
                    print(f"Warning: Error encoding {col}: {e}")
                    self.df[col + '_Encoded'] = 0
            else:
                print(f"⚠️ Column {col} not found, using default encoding")
                self.df[col + '_Encoded'] = 0
        
        # Default processing time
        self.df['Processing_Time_Hours'] = 1.0
        
        print("✅ Feature engineering completed!")
        print(f"Available columns: {list(self.df.columns)}")
        return self.df

def main():
    """Main function to run the complete project"""
    print("🚀 Starting Amazon Delivery Time Prediction Project")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AmazonDeliveryPredictor('amazon_delivery.csv')
    
    try:
        # Step 1: Data Preparation
        print("\n" + "="*50)
        print("STEP 1: DATA PREPARATION")
        print("="*50)
        df_cleaned = predictor.load_and_clean_data()
        
        if df_cleaned is None:
            print("❌ Cannot continue without data. Please check your CSV file.")
            return
        
        # Step 2: Feature Engineering
        print("\n" + "="*50)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*50)
        df_features = predictor.perform_feature_engineering()
        
        if df_features is None:
            print("❌ Feature engineering failed.")
            return
        
        print("\n🎉 PROJECT SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\n📊 DATA SUMMARY:")
        print(f"• Total records: {len(df_features)}")
        print(f"• Total columns: {len(df_features.columns)}")
        print(f"• Delivery Time range: {df_features['Delivery_Time'].min():.2f} to {df_features['Delivery_Time'].max():.2f} hours")
        print(f"• Average Delivery Time: {df_features['Delivery_Time'].mean():.2f} hours")
        
        print("\n🌐 Next steps:")
        print("1. Run 'python model_training.py' to train ML models")
        print("2. Run 'streamlit run streamlit_app.py' for web interface")
        
    except Exception as e:
        print(f"❌ Error in project execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()