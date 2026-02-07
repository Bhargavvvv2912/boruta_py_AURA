import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 🛠️ Point to the 'boruta' folder
try:
    from boruta import BorutaPy
    print("--> [✓] Successfully loaded BorutaPy from the boruta folder")
except ImportError as e:
    print(f"--> [X] CRITICAL: Could not find BorutaPy. Error: {e}")
    sys.exit(1)

def test_boruta():
    print("--- BorutaPy (Repo 30) Functional Verification ---")
    try:
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        rf = RandomForestClassifier(n_jobs=-1, max_depth=3, n_estimators=10)
        
        print("--> Running Boruta feature selection...")
        feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
        feat_selector.fit(X, y)
        
        print(f"    [✓] Success! Features selected: {feat_selector.n_features_}")
        print("--- SMOKE TEST PASSED ---")
    except AttributeError as ae:
        print(f"--> [!] EXPECTED API DRIFT: {str(ae)}")
        sys.exit(1)
    except Exception as e:
        print(f"--> [X] UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_boruta()