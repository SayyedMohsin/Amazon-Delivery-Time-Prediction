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

print("🤖 STARTING MODEL TRAINING WITH MLFLOW...")

# Create mlruns directory if not exists
os.makedirs('mlruns', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load fixed data
try:
    df = pd.read_csv('amazon_delivery_FIXED.csv')
    print(f"✅ Fixed data loaded: {df.shape}")
except FileNotFoundError:
    print("❌ Fixed data not found! Please run fixed_data_preparation.py first")
    exit()

# Prepare features
feature_columns = [
    'Distance_km', 'Agent_Age', 'Agent_Rating', 'Order_Hour', 
    'Order_DayOfWeek', 'Weather_Encoded', 'Traffic_Encoded', 
    'Vehicle_Encoded', 'Area_Encoded', 'Category_Encoded', 
    'Processing_Time_Hours', 'Is_Weekend'
]

# Use only available features
available_features = [col for col in feature_columns if col in df.columns]
print(f"📊 Using {len(available_features)} features: {available_features}")

X = df[available_features]
y = df['Delivery_Time']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"⏱️ Target range: {y.min():.2f} to {y.max():.2f} hours")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"📈 Training samples: {X_train.shape[0]}")
print(f"📊 Testing samples: {X_test.shape[0]}")

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Setup MLflow experiment
experiment_name = "Amazon_Delivery_Perfect_Prediction"
mlflow.set_experiment(experiment_name)

print(f"\n🔬 MLFLOW EXPERIMENT: {experiment_name}")
print("=" * 60)

best_model = None
best_model_name = None
best_rmse = float('inf')
results = {}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"\n🎯 TRAINING: {model_name}")
        
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
        mlflow.log_metric("training_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Store results
        results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"✅ {model_name} COMPLETED:")
        print(f"   • RMSE: {rmse:.3f} hours")
        print(f"   • MAE: {mae:.3f} hours")
        print(f"   • R²: {r2:.3f}")
        
        # Update best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name

# Save best model and info
if best_model is not None:
    joblib.dump(best_model, 'models/best_delivery_model.pkl')
    
    # Save feature list
    feature_info = {
        'features': available_features,
        'best_model': best_model_name,
        'best_rmse': best_rmse
    }
    joblib.dump(feature_info, 'models/model_info.pkl')
    
    print(f"\n🎉 BEST MODEL: {best_model_name}")
    print(f"   • RMSE: {best_rmse:.3f} hours")
    
    # Feature importance plot
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n📊 FEATURE IMPORTANCE:")
        for _, row in feature_importance.sort_values('importance', ascending=False).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")

# Save performance report
performance_df = pd.DataFrame.from_dict(results, orient='index')
performance_df[['rmse', 'mae', 'r2']].to_csv('models/model_performance.csv')
print(f"\n💾 Performance saved: models/model_performance.csv")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"📁 MLflow UI: http://127.0.0.1:5000")
print(f"📁 Best model: models/best_delivery_model.pkl")
print(f"🔧 Next: Run 'perfect_streamlit_app.py'")

# Display final results
print("\n📈 FINAL PERFORMANCE SUMMARY:")
for model_name, metrics in results.items():
    star = " ✅" if model_name == best_model_name else ""
    print(f"   {model_name}{star}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")