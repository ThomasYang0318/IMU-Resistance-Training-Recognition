import os
import runpy

TARGET = os.path.join(os.path.dirname(__file__), "apps", "realtime", "luckfox_realtime_resistance_pipeline.py")
runpy.run_path(TARGET, run_name="__main__")
