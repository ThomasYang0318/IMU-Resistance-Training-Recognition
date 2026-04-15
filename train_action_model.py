import os
import runpy

TARGET = os.path.join(os.path.dirname(__file__), "apps", "training", "train_action_model.py")
runpy.run_path(TARGET, run_name="__main__")
