import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, r2_score, mean_absolute_error
from sklearn.metrics import root_mean_squared_error

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluates classification model and measures inference time.
    Supports PyTorch MLP, XGBoost, and LightGBM models.
    """
    start_time = time.time()
    
    # Check if PyTorch MLP model
    if hasattr(model, 'forward') and hasattr(model, 'network'):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = model(X_tensor).cpu().numpy()
        preds = np.argmax(outputs, axis=1)
    else:
        # XGBoost or LightGBM
        preds = model.predict(X_test)
        # Ensure outputs are integers
        preds = np.round(preds).astype(int)
        
    end_time = time.time()
    
    inference_time_total = end_time - start_time
    inference_time_per_sample = (inference_time_total / len(X_test)) * 1000 # in ms
    
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, preds)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'inference_time_total': float(inference_time_total),
        'inference_time_per_sample': float(inference_time_per_sample),
        'predictions': preds.tolist()
    }

def evaluate_regressor(model, X_test, y_test):
    """
    Evaluates regression model and measures inference time.
    Supports PyTorch MLP, XGBoost, and LightGBM models.
    """
    start_time = time.time()
    
    # Check if PyTorch MLP model
    if hasattr(model, 'forward') and hasattr(model, 'network'):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = model(X_tensor).cpu().numpy()
        preds = outputs.flatten()
    else:
        # XGBoost or LightGBM
        preds = model.predict(X_test)
        
    end_time = time.time()
    
    inference_time_total = end_time - start_time
    inference_time_per_sample = (inference_time_total / len(X_test)) * 1000 # in ms
    
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'inference_time_total': float(inference_time_total),
        'inference_time_per_sample': float(inference_time_per_sample),
        'predictions': preds.tolist()
    }
