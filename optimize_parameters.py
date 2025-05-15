import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

from ctf4science.tune_module import ModelTuner

if __name__ == '__main__':
    ModelTuner.run_from_cli(description='Run hyperparameter tuning for the FNO model') 