"""Application configuration classes for Melbourne CBD Pedestrian Platform."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-secret-key-change-in-production")
    
    # Environment Settings
    ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = False
    TESTING = False
    
    # Data Directories & Paths
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Cloud Services Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "")
    UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN", "")
    
    # Cache Settings
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))
    
    # OSRM & Nominatim APIs
    OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "http://router.project-osrm.org/route/v1/foot")
    NOMINATIM_BASE_URL = os.getenv("NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org/search")
    
    # Default Mode
    DEFAULT_SERVER_MODE = os.getenv("SERVER_MODE", "rule")


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    ENV = "development"


class ProductionConfig(BaseConfig):
    DEBUG = False
    ENV = "production"


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    ENV = "testing"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": ProductionConfig,
}
