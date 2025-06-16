import logging
import os
from pathlib import Path
import sys
from datetime import datetime

def get_log_dir():
    if getattr(sys, "frozen", False):
        # Running from PyInstaller, if ever used
        base_dir = Path.home() / "AppData" / "Local" / "AD_APP" / "logs"
    else:
        # Local dev
        base_dir = Path.home() / "logs"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

LOG_DIR = get_log_dir()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- SYSTEM LOGGER ---
system_log_path = LOG_DIR / f"system_{timestamp}.log"
system_logger = logging.getLogger("system")
system_logger.setLevel(logging.DEBUG)
system_handler = logging.FileHandler(system_log_path, encoding="utf-8")
system_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
system_logger.addHandler(system_handler)

if not getattr(sys, "frozen", False):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    system_logger.addHandler(console_handler)
