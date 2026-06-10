import os
import uvicorn
from generate_dataset import generate_perovskite_dataset

def main():
    # Verify sample dataset is present
    dataset_path = "perovskite_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found. Generating sample perovskite dataset at {dataset_path}...")
        generate_perovskite_dataset(dataset_path)
    else:
        print(f"Sample dataset already exists at {dataset_path}.")
        
    # Ensure all directories exist
    os.makedirs(os.path.join('static', 'plots'), exist_ok=True)
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    os.makedirs(os.path.join('static', 'models'), exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Copy dataset to uploads directory so it can be loaded directly as a fallback
    shutil_dest = os.path.join('static', 'uploads', 'dataset.csv')
    try:
        import shutil
        shutil.copy(dataset_path, shutil_dest)
        print(f"Prepared sample dataset in uploads folder.")
    except Exception as e:
        print(f"Could not copy sample dataset: {e}")
        
    # Start server
    print("\nStarting comparison dashboard server at http://localhost:8000")
    print("Press Ctrl+C to terminate the process.")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == '__main__':
    main()
