import csv
import pickle
import os
from typing import Dict, Any, List
from decorators import safe_execution

def parse_csv_value(value: str) -> Any:
    """Helper to parse string values from CSV into appropriate types."""
    value = value.strip()
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

@safe_execution("Error loading parameters from CSV", return_on_error=None)
def load_params_from_csv(file_path: str) -> Dict[str, Any]:
    """Generic function to load parameters from a CSV file."""
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {k: parse_csv_value(v) for k, v in next(reader).items()}
    
@safe_execution("Error saving parameters to CSV", return_on_error=False)
def save_params_to_csv(data: Dict[str, Any], filename: str) -> bool:
    """Generic function to save a dictionary to a CSV file."""
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    print(f"Parametry zapisano do pliku: {filename}")
    return True

@safe_execution("Error saving pickle", return_on_error=False)
def save_pickle(data: Any, filename: str) -> bool:
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dane zapisano do pliku: {filename}")
    return True

@safe_execution("Error loading pickle", return_on_error=None)
def load_pickle(filename: str) -> Any:
    if not os.path.exists(filename):
        print(f"Plik {filename} nie istnieje.")
        return None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Wczytano z pliku: {filename}")
    return data