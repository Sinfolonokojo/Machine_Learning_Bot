"""
Test script to verify installation
"""

print("=" * 80)
print("TESTING INSTALLATION")
print("=" * 80)

# Test imports
try:
    import numpy as np
    print("[OK] NumPy imported successfully (version: {})".format(np.__version__))
except Exception as e:
    print("[FAIL] NumPy import failed:", e)

try:
    import pandas as pd
    print("[OK] Pandas imported successfully (version: {})".format(pd.__version__))
except Exception as e:
    print("[FAIL] Pandas import failed:", e)

try:
    import gymnasium as gym
    print("[OK] Gymnasium imported successfully (version: {})".format(gym.__version__))
except Exception as e:
    print("[FAIL] Gymnasium import failed:", e)

try:
    import matplotlib
    print("[OK] Matplotlib imported successfully (version: {})".format(matplotlib.__version__))
except Exception as e:
    print("[FAIL] Matplotlib import failed:", e)

try:
    import sklearn
    print("[OK] Scikit-learn imported successfully (version: {})".format(sklearn.__version__))
except Exception as e:
    print("[FAIL] Scikit-learn import failed:", e)

try:
    import torch
    print("[OK] PyTorch imported successfully (version: {})".format(torch.__version__))
except Exception as e:
    print("[FAIL] PyTorch import FAILED:", e)
    print("\n>>> FIX REQUIRED <<<")
    print("You need to install Microsoft Visual C++ Redistributable")
    print("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("After installing, restart your terminal and run this test again.")

try:
    import stable_baselines3
    print("[OK] Stable-Baselines3 imported successfully (version: {})".format(stable_baselines3.__version__))
except Exception as e:
    print("[FAIL] Stable-Baselines3 import FAILED:", e)
    print("(This usually works after fixing PyTorch)")

print("\n" + "=" * 80)
print("Installation test complete!")
print("=" * 80)
