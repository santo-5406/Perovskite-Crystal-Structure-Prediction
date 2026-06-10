from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import pandas as pd
import numpy as np
import time
import json
import threading
import torch
import joblib

# Import modular components
from data_preprocessing import prepare_pipeline, engineer_domain_features
from models import PerovskiteMLP, train_mlp_model, GeneticAlgorithmOptimizer, train_xgboost, train_lightgbm
from evaluation import evaluate_classifier, evaluate_regressor
from visualization import (
    plot_performance_comparison,
    plot_training_time_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importances,
    plot_nn_loss_curve,
    plot_regression_scatter,
    plot_ga_tuning_history,
    plot_hyperparameter_tuning
)

app = FastAPI(title="Materials Science ML Comparison Dashboard")

# Global training state
TRAINING_STATE = {
    'status': 'idle', # idle, running, completed, failed
    'progress': 0,
    'message': 'No training has run yet.',
    'task_type': 'classification',
    'results': {},
    'ga_history': [],
    'best_ga_params': {}
}

# Directories creation
os.makedirs('static/plots', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/models', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def create_csv_report(results, filepath):
    import csv
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Parameter/Metric', 'Value'])
        
        # Write best model info
        writer.writerow(['Summary', 'Best Performing Model', results['best_model']])
        writer.writerow(['Summary', 'Best Metric Value', f"{results['best_metric_value']:.4f}"])
        writer.writerow([])
        
        is_classification = results['is_classification']
        
        for name, data in results['models'].items():
            writer.writerow([name, '--- MODEL SUMMARY ---', ''])
            writer.writerow([name, 'Training Time (s)', f"{data['train_time']:.2f}"])
            writer.writerow([name, 'Inference Time per Sample (ms)', f"{data['metrics']['inference_time_per_sample']:.4f}"])
            
            # Hyperparameters
            for k, v in data['best_params'].items():
                writer.writerow([name, f"Hyperparameter: {k}", str(v)])
                
            # Metrics
            if is_classification:
                writer.writerow([name, 'Accuracy', f"{data['metrics']['accuracy']:.4f}"])
                writer.writerow([name, 'Precision', f"{data['metrics']['precision']:.4f}"])
                writer.writerow([name, 'Recall', f"{data['metrics']['recall']:.4f}"])
                writer.writerow([name, 'F1-score', f"{data['metrics']['f1_score']:.4f}"])
            else:
                writer.writerow([name, 'R2 score', f"{data['metrics']['r2_score']:.4f}"])
                writer.writerow([name, 'RMSE', f"{data['metrics']['rmse']:.4f}"])
                writer.writerow([name, 'MAE', f"{data['metrics']['mae']:.4f}"])
            writer.writerow([])

def run_training_in_background(filepath, target_col, is_classification, handle_imbalance):
    global TRAINING_STATE
    TRAINING_STATE = {
        'status': 'running',
        'progress': 0,
        'message': 'Initializing dataset preprocessor...',
        'task_type': 'classification' if is_classification else 'regression',
        'results': {},
        'ga_history': [],
        'best_ga_params': {}
    }
    
    try:
        # 1. Preprocess
        TRAINING_STATE['progress'] = 5
        TRAINING_STATE['message'] = "Running domain feature engineering and scaling..."
        
        X_train, X_test, y_train, y_test, scaler, le, feature_names = prepare_pipeline(
            filepath, target_col, is_classification, handle_imbalance
        )
        
        # Save preprocess modules
        joblib.dump(scaler, 'static/models/scaler.pkl')
        joblib.dump(feature_names, 'static/models/feature_names.pkl')
        if le:
            joblib.dump(le, 'static/models/label_encoder.pkl')
            
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train)) if is_classification else 1
        
        # 2. Genetic Algorithm MLP
        TRAINING_STATE['progress'] = 10
        TRAINING_STATE['message'] = "Optimizing PyTorch Neural Network using Genetic Algorithm..."
        
        def ga_callback(generation, best_fitness, best_params, tuning_history, progress_percent):
            TRAINING_STATE['progress'] = 10 + int(progress_percent * 0.4) # 10% to 50%
            # If regression, fitness is 1/(RMSE + 1e-5). Convert for display if needed
            disp_fitness = best_fitness
            TRAINING_STATE['message'] = f"GA MLP Optimization (Gen {generation}/4) | Best Val Score = {disp_fitness:.4f}"
            TRAINING_STATE['ga_history'] = tuning_history
            TRAINING_STATE['best_ga_params'] = best_params

        ga = GeneticAlgorithmOptimizer(
            input_dim=input_dim,
            output_dim=output_dim,
            is_classification=is_classification,
            pop_size=8,
            generations=4,
            epochs_per_eval=15
        )
        best_params, ga_history = ga.optimize(X_train, y_train, progress_callback=ga_callback)
        
        # 3. Retrain MLP
        TRAINING_STATE['progress'] = 50
        TRAINING_STATE['message'] = f"GA optimized Neural Network: retraining for 50 epochs..."
        
        # Split train for final validation curves
        from sklearn.model_selection import train_test_split
        if is_classification:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
            
        final_mlp = PerovskiteMLP(
            input_dim=input_dim,
            hidden_layers=best_params['neurons'],
            output_dim=output_dim,
            is_classification=is_classification,
            dropout=best_params['dropout']
        )
        
        start_t = time.time()
        train_losses, val_losses = train_mlp_model(
            final_mlp, X_tr, y_tr, X_val, y_val,
            epochs=50,
            lr=best_params['lr'],
            batch_size=32,
            is_classification=is_classification
        )
        nn_train_time = time.time() - start_t
        
        # Save NN model weights
        torch.save(final_mlp.state_dict(), 'static/models/neural_network.pt')
        joblib.dump(best_params, 'static/models/nn_best_params.pkl')
        
        # Evaluate NN
        nn_eval = evaluate_classifier(final_mlp, X_test, y_test) if is_classification else evaluate_regressor(final_mlp, X_test, y_test)
        
        # 4. XGBoost
        TRAINING_STATE['progress'] = 65
        TRAINING_STATE['message'] = "Tuning and training XGBoost model..."
        
        start_t = time.time()
        xgb_model, xgb_best_params, xgb_cv = train_xgboost(X_train, y_train, is_classification)
        xgb_train_time = time.time() - start_t
        
        joblib.dump(xgb_model, 'static/models/xgb_model.pkl')
        xgb_eval = evaluate_classifier(xgb_model, X_test, y_test) if is_classification else evaluate_regressor(xgb_model, X_test, y_test)
        
        # 5. LightGBM
        TRAINING_STATE['progress'] = 80
        TRAINING_STATE['message'] = "Tuning and training LightGBM model..."
        
        start_t = time.time()
        lgb_model, lgb_best_params, lgb_cv = train_lightgbm(X_train, y_train, is_classification)
        lgb_train_time = time.time() - start_t
        
        joblib.dump(lgb_model, 'static/models/lgb_model.pkl')
        lgb_eval = evaluate_classifier(lgb_model, X_test, y_test) if is_classification else evaluate_regressor(lgb_model, X_test, y_test)
        
        # 6. Generate comparative graphs
        TRAINING_STATE['progress'] = 90
        TRAINING_STATE['message'] = "Generating visualization heatmaps and charts..."
        
        # Delete old plots if existing
        for f in os.listdir('static/plots'):
            os.remove(os.path.join('static/plots', f))
            
        plot_nn_loss_curve(train_losses, val_losses, 'static/plots/nn_loss.png')
        plot_ga_tuning_history(ga_history, is_classification, 'static/plots/ga_tuning.png')
        plot_hyperparameter_tuning(xgb_cv, 'XGBoost', 'static/plots/xgb_tuning.png')
        
        # Comparison charts
        if is_classification:
            perf_dict = {
                'GA-NN': nn_eval['accuracy'],
                'XGBoost': xgb_eval['accuracy'],
                'LightGBM': lgb_eval['accuracy']
            }
        else:
            perf_dict = {
                'GA-NN': nn_eval['r2_score'],
                'XGBoost': xgb_eval['r2_score'],
                'LightGBM': lgb_eval['r2_score']
            }
        plot_performance_comparison(perf_dict, is_classification, 'static/plots/performance_comparison.png')
        
        time_dict = {
            'GA-NN': nn_train_time,
            'XGBoost': xgb_train_time,
            'LightGBM': lgb_train_time
        }
        plot_training_time_comparison(time_dict, 'static/plots/time_comparison.png')
        
        # Tree Importances
        plot_feature_importances(xgb_model.feature_importances_, feature_names, 'XGBoost', 'static/plots/xgb_feat_imp.png')
        plot_feature_importances(lgb_model.feature_importances_, feature_names, 'LightGBM', 'static/plots/lgb_feat_imp.png')
        
        # Task specific charts
        if is_classification:
            class_names = [str(c) for c in (le.classes_ if le else np.unique(y_test))]
            plot_confusion_matrix_heatmap(nn_eval['confusion_matrix'], class_names, 'static/plots/nn_cm.png')
            plot_confusion_matrix_heatmap(xgb_eval['confusion_matrix'], class_names, 'static/plots/xgb_cm.png')
            plot_confusion_matrix_heatmap(lgb_eval['confusion_matrix'], class_names, 'static/plots/lgb_cm.png')
        else:
            plot_regression_scatter(y_test, nn_eval['predictions'], 'static/plots/nn_scatter.png')
            plot_regression_scatter(y_test, xgb_eval['predictions'], 'static/plots/xgb_scatter.png')
            plot_regression_scatter(y_test, lgb_eval['predictions'], 'static/plots/lgb_scatter.png')
            
        # Select best model
        best_model_name = max(perf_dict, key=perf_dict.get)
        best_val = perf_dict[best_model_name]
        
        # Build training state results
        TRAINING_STATE['results'] = {
            'is_classification': is_classification,
            'best_model': best_model_name,
            'best_metric_value': best_val,
            'models': {
                'GA-NN': {
                    'metrics': {k: v for k, v in nn_eval.items() if k != 'predictions'},
                    'best_params': best_params,
                    'train_time': nn_train_time
                },
                'XGBoost': {
                    'metrics': {k: v for k, v in xgb_eval.items() if k != 'predictions'},
                    'best_params': xgb_best_params,
                    'train_time': xgb_train_time
                },
                'LightGBM': {
                    'metrics': {k: v for k, v in lgb_eval.items() if k != 'predictions'},
                    'best_params': lgb_best_params,
                    'train_time': lgb_train_time
                }
            }
        }
        
        create_csv_report(TRAINING_STATE['results'], 'static/model_comparison_report.csv')
        
        TRAINING_STATE['progress'] = 100
        TRAINING_STATE['status'] = 'completed'
        TRAINING_STATE['message'] = 'Model training and evaluation completed successfully!'
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        TRAINING_STATE['status'] = 'failed'
        TRAINING_STATE['message'] = f"Training failed: {str(e)}"

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse('templates/index.html')

@app.get("/perovskite_dataset.csv")
def get_sample_csv():
    path = "perovskite_dataset.csv"
    if not os.path.exists(path):
        path = "static/uploads/dataset.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    return FileResponse(path, media_type="text/csv", filename="perovskite_dataset.csv")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    filepath = os.path.join("static", "uploads", "dataset.csv")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Read statistics
    df = pd.read_csv(filepath)
    columns = df.columns.tolist()
    shape = df.shape
    
    # Missing values dict
    missing = df.isnull().sum().to_dict()
    
    # Head sample
    sample_data = df.head(10).replace({np.nan: None}).to_dict(orient='records')
    
    # Summary stats for numerical columns
    numerical_summary = df.describe().replace({np.nan: None}).to_dict()
    
    return {
        "filename": file.filename,
        "columns": columns,
        "shape": shape,
        "missing_values": missing,
        "sample": sample_data,
        "summary": numerical_summary
    }

@app.post("/train")
def start_training(
    background_tasks: BackgroundTasks,
    target_column: str = Form(...),
    task_type: str = Form(...),
    handle_imbalance: bool = Form(True)
):
    global TRAINING_STATE
    filepath = os.path.join("static", "uploads", "dataset.csv")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Please upload a dataset first.")
        
    is_classification = (task_type == 'classification')
    
    # Verify target exists
    df = pd.read_csv(filepath)
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")
        
    # Run in standard thread to avoid blockages
    thread = threading.Thread(
        target=run_training_in_background,
        args=(filepath, target_column, is_classification, handle_imbalance)
    )
    thread.start()
    
    return {"status": "started", "message": "Training pipeline initiated."}

@app.get("/stream-progress")
def stream_progress():
    def event_generator():
        while True:
            # Yield structured training updates
            data = json.dumps({
                'status': TRAINING_STATE['status'],
                'progress': TRAINING_STATE['progress'],
                'message': TRAINING_STATE['message'],
                'task_type': TRAINING_STATE['task_type'],
                'results': TRAINING_STATE['results'],
                'ga_history': TRAINING_STATE['ga_history'],
                'best_ga_params': TRAINING_STATE['best_ga_params']
            })
            yield f"data: {data}\n\n"
            if TRAINING_STATE['status'] in ['completed', 'failed']:
                break
            time.sleep(1.0)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/results")
def get_results():
    return TRAINING_STATE

@app.get("/download-report")
def download_report():
    filepath = 'static/model_comparison_report.csv'
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="No report is available yet. Please run training first.")
    return FileResponse(filepath, media_type="text/csv", filename="model_comparison_report.csv")

@app.post("/predict")
def predict(data: dict):
    """
    Accepts raw features dictionary, applies scaling, engineers domain features, and queries models.
    """
    # Load scaling and models
    try:
        scaler = joblib.load('static/models/scaler.pkl')
        feature_names = joblib.load('static/models/feature_names.pkl')
        xgb_model = joblib.load('static/models/xgb_model.pkl')
        lgb_model = joblib.load('static/models/lgb_model.pkl')
        
        is_classification = TRAINING_STATE['task_type'] == 'classification'
        le = None
        if is_classification and os.path.exists('static/models/label_encoder.pkl'):
            le = joblib.load('static/models/label_encoder.pkl')
            
        # Convert user query data to df for domain engineering
        df_input = pd.DataFrame([data])
        
        # Engineer domain features
        df_input = engineer_domain_features(df_input)
        
        # Match input to trained columns (handling missing columns via dummy flags)
        # We need to one-hot encode inputs similarly
        # To handle this robustly, we create a zero-filled dataframe with all training features
        # and populate it with matching features from our input.
        df_model_input = pd.DataFrame(0.0, index=[0], columns=feature_names)
        
        # Fill in numerical features
        for col in df_input.columns:
            if col in df_model_input.columns:
                df_model_input[col] = float(df_input[col].iloc[0])
            elif col in ['element_A', 'element_B', 'element_X']:
                # Handle categorical one-hot matching
                val = df_input[col].iloc[0]
                dummy_col = f"{col}_{val}"
                if dummy_col in df_model_input.columns:
                    df_model_input[dummy_col] = 1.0
                    
        # Scale numerical inputs
        X_scaled = scaler.transform(df_model_input.values)
        
        # Run XGBoost & LGBM
        xgb_pred = xgb_model.predict(X_scaled)[0]
        lgb_pred = lgb_model.predict(X_scaled)[0]
        
        # Run PyTorch NN
        # Load best params to get layer structure
        best_params = joblib.load('static/models/nn_best_params.pkl')
        input_dim = len(feature_names)
        output_dim = len(le.classes_) if is_classification and le else 1
        
        nn_model = PerovskiteMLP(
            input_dim=input_dim,
            hidden_layers=best_params['neurons'],
            output_dim=output_dim,
            is_classification=is_classification
        )
        nn_model.load_state_dict(torch.load('static/models/neural_network.pt', map_location=torch.device('cpu')))
        nn_model.eval()
        
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32)
            nn_out = nn_model(x_t).numpy()
            
        if is_classification:
            nn_pred_idx = np.argmax(nn_out, axis=1)[0]
            if le:
                nn_pred = str(le.inverse_transform([nn_pred_idx])[0])
                xgb_pred_lbl = str(le.inverse_transform([int(round(xgb_pred))])[0])
                lgb_pred_lbl = str(le.inverse_transform([int(round(lgb_pred))])[0])
            else:
                nn_pred = int(nn_pred_idx)
                xgb_pred_lbl = int(round(xgb_pred))
                lgb_pred_lbl = int(round(lgb_pred))
        else:
            nn_pred = float(nn_out.flatten()[0])
            xgb_pred_lbl = float(xgb_pred)
            lgb_pred_lbl = float(lgb_pred)
            
        return {
            "GA-NN": nn_pred,
            "XGBoost": xgb_pred_lbl,
            "LightGBM": lgb_pred_lbl,
            "engineered_features": df_input.iloc[0].to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference prediction failed: {str(e)}")
