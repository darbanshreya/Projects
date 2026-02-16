"""
Design2Wear-AI — First Time Setup
==================================
Run this ONCE to generate all data files needed by the notebook and app.

Usage:
    python setup.py
"""

import os
import sys

print("=" * 60)
print("Design2Wear-AI — First Time Setup")
print("=" * 60)

# Check dependencies
try:
    import pandas as pd
    import numpy as np
    import sklearn
    print("✅ Core libraries found")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Check for Kaggle dataset
try:
    import kagglehub
    print("✅ kagglehub found")
    print("📥 Downloading Kaggle dataset (this may take a few minutes)...")
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    print(f"✅ Dataset downloaded to: {path}")
except Exception as e:
    print(f"⚠️  Could not download dataset: {e}")
    print("   Download manually from:")
    print("   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
    path = None

# Check for TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} found")
except ImportError:
    print("⚠️  TensorFlow not found. Install with: pip install tensorflow")
    print("   (Needed for model training cells in notebook)")

# Check for Streamlit
try:
    import streamlit
    print(f"✅ Streamlit found")
except ImportError:
    print("⚠️  Streamlit not found. Install with: pip install streamlit")

print("\n" + "=" * 60)
print("Setup complete! Next steps:")
print("=" * 60)
print("1. Open Design2WearAI_Clean.ipynb in Jupyter")
print("2. Run all cells top to bottom (generates data files)")
print("3. Then run: streamlit run app.py")
print("=" * 60)
