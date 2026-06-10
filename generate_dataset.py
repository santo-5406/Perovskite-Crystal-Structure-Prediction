import pandas as pd
import numpy as np
import os

def generate_perovskite_dataset(output_path="perovskite_dataset.csv", num_samples=800):
    np.random.seed(42)
    
    # Base elements details (ionic radius in Angstroms, Pauling electronegativity)
    a_elements = {
        'Sr': {'radius': 1.44, 'en': 0.95},
        'Ba': {'radius': 1.61, 'en': 0.89},
        'Ca': {'radius': 1.34, 'en': 1.00},
        'La': {'radius': 1.36, 'en': 1.10},
        'Pb': {'radius': 1.49, 'en': 2.33},
        'Na': {'radius': 1.39, 'en': 0.93},
        'K':  {'radius': 1.64, 'en': 0.82}
    }
    
    b_elements = {
        'Ti': {'radius': 0.605, 'en': 1.54},
        'Zr': {'radius': 0.72,  'en': 1.33},
        'Mn': {'radius': 0.67,  'en': 1.55},
        'Fe': {'radius': 0.645, 'en': 1.83},
        'Co': {'radius': 0.61,  'en': 1.88},
        'Sn': {'radius': 0.69,  'en': 1.96},
        'Nb': {'radius': 0.64,  'en': 1.60},
        'Ta': {'radius': 0.64,  'en': 1.50}
    }
    
    x_elements = {
        'O':  {'radius': 1.40, 'en': 3.44},
        'F':  {'radius': 1.33, 'en': 3.98},
        'Cl': {'radius': 1.81, 'en': 3.16}
    }
    
    records = []
    
    a_keys = list(a_elements.keys())
    b_keys = list(b_elements.keys())
    x_keys = list(x_elements.keys())
    
    for _ in range(num_samples):
        # Select base elements
        a = np.random.choice(a_keys)
        b = np.random.choice(b_keys)
        x = np.random.choice(x_keys)
        
        # Base properties
        r_A = a_elements[a]['radius']
        en_A = a_elements[a]['en']
        
        r_B = b_elements[b]['radius']
        en_B = b_elements[b]['en']
        
        r_X = x_elements[x]['radius']
        en_X = x_elements[x]['en']
        
        # Add experimental variations (doping, temperature, pressure)
        doping_level = np.random.uniform(0.0, 0.25)
        dopant_radius = np.random.uniform(1.2, 1.7) # Dopant at A-site
        dopant_en = np.random.uniform(0.8, 2.0)
        
        # Effective A-site properties
        r_A_eff = (1 - doping_level) * r_A + doping_level * dopant_radius
        en_A_eff = (1 - doping_level) * en_A + doping_level * dopant_en
        
        # Environmental conditions
        temperature = np.random.uniform(100.0, 1000.0) # in Kelvin
        pressure = np.random.uniform(0.0, 12.0) # in GPa
        
        # Calculate Goldschmidt tolerance factor (before preprocessing to simulate it)
        t = (r_A_eff + r_X) / (np.sqrt(2) * (r_B + r_X))
        
        # Determine crystal structure based on tolerance factor and env conditions
        # Temperature stabilizes Cubic phase; pressure stabilizes denser distorted phases
        t_eff = t + 0.00015 * (temperature - 300) - 0.002 * pressure
        
        if 0.93 <= t_eff <= 1.02:
            crystal_structure = 'Cubic'
        elif 1.02 < t_eff <= 1.09:
            crystal_structure = 'Tetragonal'
        elif 0.86 <= t_eff < 0.93:
            crystal_structure = 'Rhombohedral'
        else:
            crystal_structure = 'Orthorhombic'
            
        # Introduce slight noise in classification (to simulate experimental issues)
        if np.random.rand() < 0.03:
            crystal_structure = np.random.choice(['Cubic', 'Tetragonal', 'Rhombohedral', 'Orthorhombic'])
            
        # Calculate Lattice Parameter 'a' (in Angstroms) based on physical size
        # a should increase with larger ionic sizes and temperature, and decrease with pressure
        lattice_parameter = 0.55 * np.sqrt(2) * (r_A_eff + r_X) + 0.85 * (r_B + r_X) + 0.00012 * temperature - 0.008 * pressure
        # Add experimental variance noise
        lattice_parameter += np.random.normal(0, 0.015)
        lattice_parameter = round(lattice_parameter, 4)
        
        # Create formula
        if doping_level > 0.01:
            formula = f"{a}1-{doping_level:.2f}Dop{doping_level:.2f}{b}{x}3"
        else:
            formula = f"{a}{b}{x}3"
            
        records.append({
            'formula': formula,
            'element_A': a,
            'element_B': b,
            'element_X': x,
            'r_A': round(r_A_eff, 4),
            'r_B': r_B,
            'r_X': r_X,
            'electronegativity_A': round(en_A_eff, 4),
            'electronegativity_B': en_B,
            'electronegativity_X': en_X,
            'temperature_K': round(temperature, 1),
            'pressure_GPa': round(pressure, 2),
            'crystal_structure': crystal_structure,
            'lattice_parameter': lattice_parameter
        })
        
    df = pd.DataFrame(records)
    
    # Introduce some missing values (around 2% missing in r_A and electronegativity_A) to test preprocessor imputation
    missing_indices_r = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
    missing_indices_en = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
    df.loc[missing_indices_r, 'r_A'] = np.nan
    df.loc[missing_indices_en, 'electronegativity_A'] = np.nan
    
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} with {len(df)} records.")
    return df

if __name__ == '__main__':
    generate_perovskite_dataset()
