import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

try:
    from boruta import BorutaPy
    print("--> [✓] Loaded BorutaPy")
except ImportError:
    print("CRITICAL: Boruta folder not found.")
    sys.exit(1)

def test_boruta():
    print("--- BorutaPy API Stress Test ---")
    try:
        # Create a more complex dataset (20 features, 5 actually matter)
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)
        
        # Use a real RF
        rf = RandomForestClassifier(n_jobs=-1, max_depth=5)
        
        # Force a high number of trials to ensure we hit the internal logic
        print("--> Running Boruta (Max 20 iterations)...")
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, max_iter=20, random_state=1)
        feat_selector.fit(X, y)
        
        print(f"    [✓] Success! Features confirmed: {np.sum(feat_selector.support_)}")
        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        # Check if it's the famous 'AttributeError' related to scikit-learn tags
        print(f"--> [!] CAUGHT FAILURE: {type(e).__name__}: {str(e)}")
        # If it fails, we return 1 so the 'Upgrade' pass reflects the break
        sys.exit(1)

if __name__ == "__main__":
    test_boruta()