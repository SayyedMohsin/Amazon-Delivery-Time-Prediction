import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import os

print("🤖 STARTING AUTOMATIC MODEL TRAINING...")

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('mlruns', exist_ok=True)

# Load prepared data
try:
    df = pd.read_csv('amazon_delivery_FIXED.csv')
    print(f"✅ Prepared data loaded: {df.shape}")
except FileNotFoundError:
    print("❌ Prepared data not found! Please run fixed_data_preparation.py first")
    exit()

# Prepare features
feature_columns = [
    'Distance_km', 'Agent_Age', 'Agent_Rating', 'Order_Hour', 
    'Order_DayOfWeek', 'Weather_Encoded', 'Traffic_Encoded', 
    'Vehicle_Encoded', 'Area_Encoded', 'Category_Encoded', 
    'Processing_Time_Hours', 'Is_Weekend'
]

available_features = [col for col in feature_columns if col in df.columns]
print(f"📊 Using {len(available_features)} features")

X = df[available_features]
y = df['Delivery_Time']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"⏱️ Delivery Time: {y.min():.1f} to {y.max():.1f} hours (avg: {y.mean():.1f})")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📈 Training samples: {X_train.shape[0]}")
print(f"📊 Testing samples: {X_test.shape[0]}")

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Setup MLflow
mlflow.set_experiment("Amazon_Delivery_Prediction")

print(f"\n🔬 TRAINING MODELS...")
print("=" * 60)

best_model = None
best_model_name = None
best_rmse = float('inf')
results = {}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"\n🎯 Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Store results
        results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model': model
        }
        
        print(f"✅ {model_name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
        
        # Update best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name

# Save best model
if best_model is not None:
    joblib.dump(best_model, 'models/best_delivery_model.pkl')
    
    # Save model info
    model_info = {
        'features': available_features,
        'best_model': best_model_name,
        'best_rmse': best_rmse,
        'model_performance': results
    }
    joblib.dump(model_info, 'models/model_info.pkl')
    
    print(f"\n🎉 BEST MODEL: {best_model_name}")
    print(f"   • RMSE: {best_rmse:.3f} hours")
    print(f"   • Model saved: models/best_delivery_model.pkl")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 TOP FEATURES:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")

print("\n" + "=" * 60)
print("🎯 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"📊 View results: mlflow ui")
print(f"🌐 Launch app: streamlit run perfect_streamlit_app.py")
