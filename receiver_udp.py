import os
import runpy

TARGET = os.path.join(os.path.dirname(__file__), "apps", "visualization", "receiver_udp.py")
runpy.run_path(TARGET, run_name="__main__")
