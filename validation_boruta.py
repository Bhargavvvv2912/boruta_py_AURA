import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 🛠️ THE PATH HACK: Force Python to look in the root directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    # This matches the filename 'boruta_py.py' in the original repo
    from boruta_py import BorutaPy
    print("--> [✓] Successfully loaded BorutaPy from boruta_py.py")
except ImportError as e:
    print(f"--> [X] CRITICAL: Could not find boruta_py.py. Error: {e}")
    print(f"Files present: {os.listdir('.')}")
    sys.exit(1)

def test_boruta():
    print("--- BorutaPy (Repo 30) Functional Verification ---")
    try:
        # Create tiny dummy data
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        
        # Initialize a standard Random Forest
        rf = RandomForestClassifier(n_jobs=-1, max_depth=3, n_estimators=10)
        
        # Attempt to fit Boruta
        print("--> Running Boruta feature selection...")
        feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
        feat_selector.fit(X, y)
        
        print(f"    [✓] Success! Features selected: {feat_selector.n_features_}")
        print("--- SMOKE TEST PASSED ---")

    except AttributeError as ae:
        print(f"--> [!] EXPECTED API DRIFT ERROR: {str(ae)}")
        # We exit with 1 because the 'Upgrade' pass SHOULD fail this way
        sys.exit(1)
    except Exception as e:
        print(f"--> [X] UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_boruta()