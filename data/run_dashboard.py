import os
import sys
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
app = os.path.join(here, "app.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", app])