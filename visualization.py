import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Create static dir if it doesn't exist
os.makedirs(os.path.join('static', 'plots'), exist_ok=True)

# Custom aesthetics matching our premium dark mode glassmorphism dashboard
def apply_custom_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#0f172a' # Slate 900
    plt.rcParams['axes.facecolor'] = '#1e293b'   # Slate 800
    plt.rcParams['grid.color'] = '#334155'       # Slate 700
    plt.rcParams['text.color'] = '#f8fafc'       # Slate 50
    plt.rcParams['axes.labelcolor'] = '#cbd5e1'  # Slate 300
    plt.rcParams['xtick.color'] = '#94a3b8'      # Slate 400
    plt.rcParams['ytick.color'] = '#94a3b8'      # Slate 400
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11

def plot_performance_comparison(metrics_dict, is_classification, save_path):
    """
    metrics_dict: e.g. {'GA-NN': 0.85, 'XGBoost': 0.89, 'LightGBM': 0.88}
    """
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    models = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Elegant gradient-like colors
    colors = ['#6366f1', '#3b82f6', '#10b981'] # Indigo, Blue, Emerald
    
    bars = ax.bar(models, values, color=colors[:len(models)], width=0.5, edgecolor='#475569', linewidth=1)
    
    metric_name = "Accuracy" if is_classification else "$R^2$ Score"
    ax.set_title(f"Model Performance Comparison ({metric_name})", pad=15)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, max(max(values) * 1.15, 1.0) if is_classification else max(values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#e2e8f0', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_training_time_comparison(training_times_dict, save_path):
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    models = list(training_times_dict.keys())
    values = list(training_times_dict.values())
    
    colors = ['#a855f7', '#ec4899', '#f43f5e'] # Purple, Pink, Rose
    
    bars = ax.bar(models, values, color=colors[:len(models)], width=0.5, edgecolor='#475569', linewidth=1)
    
    ax.set_title("Model Training Time Comparison (Seconds)", pad=15)
    ax.set_ylabel("Time (seconds)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#e2e8f0', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_confusion_matrix_heatmap(cm, class_names, save_path):
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = np.array(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Tick formatting
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix Heatmap",
           ylabel="True Label",
           xlabel="Predicted Label")
           
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Write values in heatmap cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_feature_importances(importances, feature_names, model_name, save_path, max_features=10):
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:max_features]
    
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Use horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, align='center', color='#0284c7', edgecolor='#475569', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel('Relative Importance')
    ax.set_title(f'{model_name} Top Feature Importances', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_nn_loss_curve(train_losses, val_losses, save_path):
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(train_losses, label='Train Loss', color='#6366f1', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', color='#f43f5e', linewidth=2, linestyle='--')
    
    ax.set_title("Neural Network Learning Curves (Loss vs Epochs)", pad=15)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, facecolor='#1e293b', edgecolor='#475569')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_regression_scatter(y_true, y_pred, save_path):
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    # Draw scatter of points
    ax.scatter(y_true, y_pred, alpha=0.6, color='#38bdf8', edgecolors='#0284c7', s=25)
    
    # Reference perfect fit line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit', linewidth=1.5)
    
    # Stats annotation
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    textstr = '\n'.join((
        f'$R^2$: {r2:.4f}',
        f'RMSE: {rmse:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='#475569')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.set_title("Regression Predicted vs Actual Scatter Plot", pad=15)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend(loc='lower right', frameon=True, facecolor='#1e293b', edgecolor='#475569')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_ga_tuning_history(tuning_history, is_classification, save_path):
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    
    generations = [item['generation'] for item in tuning_history]
    best_fitness = [item['best_fitness'] for item in tuning_history]
    avg_fitness = [item['avg_fitness'] for item in tuning_history]
    
    # If regression, we displayed fitness as 1/(RMSE + 1e-5). Let's convert to RMSE for meaningful plots
    # Wait, the history from GA optimizer saves best_fitness. Let's trace it and plot fitness directly.
    ax.plot(generations, best_fitness, marker='o', label='Best Fitness', color='#10b981', linewidth=2)
    ax.plot(generations, avg_fitness, marker='s', label='Avg Fitness', color='#3b82f6', linewidth=2, linestyle='--')
    
    metric_label = "Accuracy" if is_classification else "Fitness (1/RMSE)"
    ax.set_title(f"Genetic Algorithm Hyperparameter Optimization History", pad=15)
    ax.set_xlabel("Generations")
    ax.set_ylabel(metric_label)
    ax.set_xticks(generations)
    ax.legend(frameon=True, facecolor='#1e293b', edgecolor='#475569')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_hyperparameter_tuning(cv_results, model_name, save_path):
    """
    Plots training scores vs parameters from GridSearchCV
    """
    apply_custom_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Extract mean test score and parameters
    mean_scores = cv_results['mean_test_score']
    # If the score is negative (e.g., neg_mean_squared_error), convert to positive RMSE for plot
    is_neg = False
    if len(mean_scores) > 0 and mean_scores[0] < 0:
        mean_scores = np.sqrt(-mean_scores)
        is_neg = True
        
    ax.plot(mean_scores, marker='x', color='#f43f5e', linewidth=1.5, label='Cross-Val Score')
    
    ax.set_title(f"{model_name} GridSearchCV Tuning Performance", pad=15)
    ax.set_xlabel("Parameter Combinations")
    ax.set_ylabel("Validation RMSE" if is_neg else "Validation Accuracy")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
