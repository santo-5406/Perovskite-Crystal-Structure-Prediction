import numpy as np
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import os

# Suppress lightgbm warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# =====================================================================
# 1. PYTORCH MLP MODEL
# =====================================================================

class PerovskiteMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, is_classification, dropout=0.1):
        super(PerovskiteMLP, self).__init__()
        self.is_classification = is_classification
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_mlp_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.005, batch_size=32, is_classification=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Convert data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    if is_classification:
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        criterion = nn.MSELoss()
        
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(X_train)
        train_losses.append(epoch_loss)
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_x, val_y = X_val_t.to(device), y_val_t.to(device)
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            val_losses.append(val_loss)
            
        scheduler.step(val_loss)
        
    return train_losses, val_losses

# =====================================================================
# 2. GENETIC ALGORITHM OPTIMIZER FOR NN
# =====================================================================

class GeneticAlgorithmOptimizer:
    def __init__(self, input_dim, output_dim, is_classification, pop_size=8, generations=4, epochs_per_eval=15):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_classification = is_classification
        self.pop_size = pop_size
        self.generations = generations
        self.epochs_per_eval = epochs_per_eval
        
        # Hyperparameter search space definitions
        self.layer_options = [1, 2, 3]
        self.neuron_options = [16, 32, 64, 128]
        self.lr_range = (0.0005, 0.02)
        self.dropout_range = (0.0, 0.3)

    def _create_random_chromosome(self):
        num_layers = random.choice(self.layer_options)
        neurons = [random.choice(self.neuron_options) for _ in range(num_layers)]
        lr = random.uniform(*self.lr_range)
        dropout = random.uniform(*self.dropout_range)
        return {
            'num_layers': num_layers,
            'neurons': neurons,
            'lr': lr,
            'dropout': dropout
        }

    def _calculate_fitness(self, chromosome, X_train, y_train, X_val, y_val):
        """
        Evaluate fitness by training a small network and returning validation performance.
        Higher is better.
        """
        set_seed(42) # Consistent evaluation seed
        
        model = PerovskiteMLP(
            input_dim=self.input_dim,
            hidden_layers=chromosome['neurons'],
            output_dim=self.output_dim,
            is_classification=self.is_classification,
            dropout=chromosome['dropout']
        )
        
        try:
            train_losses, val_losses = train_mlp_model(
                model, X_train, y_train, X_val, y_val,
                epochs=self.epochs_per_eval,
                lr=chromosome['lr'],
                batch_size=32,
                is_classification=self.is_classification
            )
            
            # Evaluate model performance on validation set
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                outputs = model(X_val_t).cpu().numpy()
                
            if self.is_classification:
                preds = np.argmax(outputs, axis=1)
                acc = accuracy_score(y_val, preds)
                fitness = acc # High accuracy is better
            else:
                preds = outputs.flatten()
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                fitness = 1.0 / (rmse + 1e-5) # Low RMSE is better -> higher fitness
                
            return fitness, val_losses[-1]
        except Exception as e:
            print(f"Error training chromosome {chromosome}: {e}")
            return 0.0, 999.0

    def _crossover(self, parent1, parent2):
        # Determine number of layers
        num_layers = random.choice([parent1['num_layers'], parent2['num_layers']])
        
        # Mix neurons
        neurons = []
        for i in range(num_layers):
            p1_neurons = parent1['neurons'][i] if i < len(parent1['neurons']) else random.choice(self.neuron_options)
            p2_neurons = parent2['neurons'][i] if i < len(parent2['neurons']) else random.choice(self.neuron_options)
            neurons.append(random.choice([p1_neurons, p2_neurons]))
            
        # Float features blend
        alpha = random.random()
        lr = alpha * parent1['lr'] + (1 - alpha) * parent2['lr']
        dropout = alpha * parent1['dropout'] + (1 - alpha) * parent2['dropout']
        
        return {
            'num_layers': num_layers,
            'neurons': neurons,
            'lr': np.clip(lr, *self.lr_range),
            'dropout': np.clip(dropout, *self.dropout_range)
        }

    def _mutate(self, chromosome):
        mutated = copy.deepcopy(chromosome)
        
        # Mutate number of layers
        if random.random() < 0.2:
            mutated['num_layers'] = random.choice(self.layer_options)
            # Adjust neurons list
            if len(mutated['neurons']) < mutated['num_layers']:
                mutated['neurons'] = mutated['neurons'] + [random.choice(self.neuron_options) for _ in range(mutated['num_layers'] - len(mutated['neurons']))]
            else:
                mutated['neurons'] = mutated['neurons'][:mutated['num_layers']]
                
        # Mutate neurons
        for i in range(len(mutated['neurons'])):
            if random.random() < 0.2:
                mutated['neurons'][i] = random.choice(self.neuron_options)
                
        # Mutate learning rate
        if random.random() < 0.2:
            mutated['lr'] = np.clip(mutated['lr'] + random.normalvariate(0, 0.005), *self.lr_range)
            
        # Mutate dropout
        if random.random() < 0.2:
            mutated['dropout'] = np.clip(mutated['dropout'] + random.normalvariate(0, 0.05), *self.dropout_range)
            
        return mutated

    def optimize(self, X_train, y_train, progress_callback=None):
        """
        Runs the GA optimization.
        `progress_callback` signature: callback(generation, best_fitness, best_params, tuning_history, progress_percent)
        """
        # Split train into train and val for fitness evaluations
        from sklearn.model_selection import train_test_split
        if self.is_classification:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
        # Initialize population
        population = [self._create_random_chromosome() for _ in range(self.pop_size)]
        tuning_history = []
        best_chromosome = None
        best_fitness = -1.0
        
        total_steps = self.generations * self.pop_size
        step_counter = 0
        
        for gen in range(self.generations):
            fitnesses = []
            
            # Evaluate all individuals
            for ind_idx, chromosome in enumerate(population):
                fitness, last_val_loss = self._calculate_fitness(chromosome, X_tr, y_tr, X_val, y_val)
                fitnesses.append(fitness)
                
                # Check for overall best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_chromosome = copy.deepcopy(chromosome)
                    
                step_counter += 1
                progress_percent = (step_counter / total_steps) * 100
                
                # Report intermediate status
                if progress_callback:
                    progress_callback(
                        generation=gen + 1,
                        best_fitness=best_fitness if self.is_classification else (1.0 / best_fitness - 1e-5), # Convert back if regression
                        best_params=best_chromosome,
                        tuning_history=tuning_history,
                        progress_percent=progress_percent
                    )
                    
            # Record gen metrics
            avg_fitness = np.mean(fitnesses)
            tuning_history.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_params': copy.deepcopy(best_chromosome)
            })
            
            # Next gen mating pool selection (Tournament)
            next_population = [best_chromosome] # Elitism
            while len(next_population) < self.pop_size:
                # Tournament selection
                tournament = random.sample(list(zip(population, fitnesses)), 3)
                tournament.sort(key=lambda x: x[1], reverse=True)
                p1 = tournament[0][0]
                
                tournament = random.sample(list(zip(population, fitnesses)), 3)
                tournament.sort(key=lambda x: x[1], reverse=True)
                p2 = tournament[0][0]
                
                # Crossover
                child = self._crossover(p1, p2)
                # Mutation
                child = self._mutate(child)
                next_population.append(child)
                
            population = next_population
            
        return best_chromosome, tuning_history

# =====================================================================
# 3. XGBOOST & LIGHTGBM MODELS
# =====================================================================

def train_xgboost(X_train, y_train, is_classification=True):
    """
    Fits and tunes an XGBoost model using GridSearchCV.
    """
    if is_classification:
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        scoring = 'accuracy'
    else:
        model = xgb.XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        scoring = 'neg_mean_squared_error'
        
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.cv_results_

def train_lightgbm(X_train, y_train, is_classification=True):
    """
    Fits and tunes a LightGBM model using GridSearchCV.
    """
    if is_classification:
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        scoring = 'accuracy'
    else:
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        scoring = 'neg_mean_squared_error'
        
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.cv_results_

if __name__ == '__main__':
    # Simple check script to ensure code runs and imports correctly
    print("Testing ML modules...")
    X = np.random.randn(100, 10)
    y_c = np.random.randint(0, 3, size=100)
    y_r = np.random.randn(100)
    
    print("Testing PyTorch setup...")
    mlp = PerovskiteMLP(10, [32, 16], 3, is_classification=True)
    print(mlp)
    
    print("\nTesting GA Optimizer (Classification)...")
    ga = GeneticAlgorithmOptimizer(10, 3, is_classification=True, pop_size=4, generations=2, epochs_per_eval=2)
    best_chrom, history = ga.optimize(X, y_c, progress_callback=lambda **kwargs: print(f"GA Progress: {kwargs['progress_percent']:.1f}%"))
    print(f"GA Best Chromosome found: {best_chrom}")
    
    print("\nTesting XGBoost training (Classification)...")
    xgb_m, xgb_p, _ = train_xgboost(X, y_c, is_classification=True)
    print(f"XGBoost parameters: {xgb_p}")
    
    print("\nTesting LightGBM training (Regression)...")
    lgb_m, lgb_p, _ = train_lightgbm(X, y_r, is_classification=False)
    print(f"LightGBM parameters: {lgb_p}")
    print("All models verify successfully!")
