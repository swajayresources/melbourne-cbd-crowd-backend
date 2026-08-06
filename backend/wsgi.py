"""Production WSGI entrypoint for Gunicorn and Render Web Service deployment."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure backend directory is prioritized on Python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) in sys.path:
    sys.path.remove(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR))

# Filter out Render's parent project directory if it collides with 'src' package
for p in list(sys.path[1:]):
    if p.endswith("/src") or p.endswith("\\src"):
        if not (p.endswith("backend/src") or p.endswith("backend\\src")):
            sys.path.remove(p)

from app import create_app

env_name = os.getenv("FLASK_ENV", "production")
app = create_app(env_name)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=(env_name == "development"))
