"""Production WSGI entrypoint for Gunicorn and Render Web Service deployment."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on Python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app

env_name = os.getenv("FLASK_ENV", "production")
app = create_app(env_name)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=(env_name == "development"))
