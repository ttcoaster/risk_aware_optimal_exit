import sys
import numpy as np
import pandas as pd
import gymnasium as gym
import stable_baselines3 as sb3
import torch
import FinanceDataReader as fdr
import vectorbt as vbt
import quantstats as qs

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Gymnasium:", gym.__version__)
print("SB3:", sb3.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("FDR:", getattr(fdr, "__version__", "import OK"))
print("VectorBT:", vbt.__version__)
print("QuantStats:", getattr(qs, "__version__", "import OK"))

print("All imports passed")