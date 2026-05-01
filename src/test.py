import pandas as pd
import numpy as np
import torch

print("pandas :", pd.__version__)
print("numpy  :", np.__version__)
print("torch  :", torch.__version__)
print("CUDA dispo :", torch.cuda.is_available())