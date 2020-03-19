"""File with configs"""
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_PATH, "models")
DATA_PATH = os.path.join(ROOT_PATH, "data")
TMP_PATH = os.path.join(ROOT_PATH, "tmp")
LOG_PATH = os.path.join(ROOT_PATH, "logs")
