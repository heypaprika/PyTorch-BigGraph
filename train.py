# train.py

import os
import sys
from torchbiggraph.config import parse_config
from torchbiggraph.train import main as train_main
import torch, torchbiggraph
from torchbiggraph import _C

print("PYTHON:", __import__("sys").version)
print("TORCH:", torch.__version__)
print("CUDA avail:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("torchbiggraph:", torchbiggraph.__file__)
print("_C module:", _C.__file__)

def main():
    config_path = os.environ.get("CONFIG_PATH", "fb15k_config_gpu_sagemaker.py")
    sys.argv = ["train.py", config_path]

    train_main()

if __name__ == "__main__":
    main()
