"""Upstash Redis and in-memory fallback cache service."""
from __future__ import annotations

import json
import time
from typing import Any, Optional
import urllib.request
import urllib.error


class CacheService:
    def __init__(self, redis_url: str = "", redis_token: str = "", default_ttl: int = 60):
        self.redis_url = redis_url.rstrip("/")
        self.redis_token = redis_token
        self.default_ttl = default_ttl
        self._memory_cache: dict[str, tuple[Any, float]] = {}

    def is_redis_configured(self) -> bool:
        return bool(self.redis_url and self.redis_token)

    def get(self, key: str) -> Optional[Any]:
        # Try Upstash Redis if configured
        if self.is_redis_configured():
            try:
                url = f"{self.redis_url}/get/{key}"
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {self.redis_token}"}
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    result = data.get("result")
                    if result:
                        return json.loads(result)
            except Exception:
                pass  # Fallback to local in-memory cache on error

        # In-memory fallback
        if key in self._memory_cache:
            val, expire_at = self._memory_cache[key]
            if time.time() < expire_at:
                return val
            else:
                del self._memory_cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        ttl_sec = ttl if ttl is not None else self.default_ttl
        json_val = json.dumps(value)

        # In-memory save
        self._memory_cache[key] = (value, time.time() + ttl_sec)

        # Try Upstash Redis if configured
        if self.is_redis_configured():
            try:
                url = f"{self.redis_url}/set/{key}?EX={ttl_sec}"
                req = urllib.request.Request(
                    url,
                    data=json_val.encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.redis_token}",
                        "Content-Type": "text/plain",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    return res_data.get("result") == "OK"
            except Exception:
                pass
        return True

    def clear_expired(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._memory_cache.items() if now >= exp]
        for k in expired:
            del self._memory_cache[k]
