# train.py

import os
import sys
from torchbiggraph.config import parse_config
from torchbiggraph.train import main as train_main

def main():
    config_path = os.environ.get("CONFIG_PATH", "fb15k_config_gpu_sagemaker.py")
    sys.argv = ["train.py", config_path]

    train_main()

if __name__ == "__main__":
    main()
