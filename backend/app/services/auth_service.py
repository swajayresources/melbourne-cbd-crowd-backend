"""Zero-PII Anonymous Authentication Service (APP / Privacy Act 1988 compliant)."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Any, Dict, Optional


class AuthService:
    def __init__(self, secret_key: str = "zero-pii-secret"):
        self.secret_key = secret_key

    def _b64url_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _b64url_decode(self, data_str: str) -> bytes:
        padding = "=" * (4 - (len(data_str) % 4))
        return base64.urlsafe_b64decode((data_str + padding).encode("utf-8"))

    def hash_public_key(self, public_key_pem_or_jwk: str) -> str:
        """Derive an anonymous hash ID from the client's WebCrypto ECDSA key."""
        clean_key = public_key_pem_or_jwk.strip().encode("utf-8")
        return hashlib.sha256(clean_key).hexdigest()[:32]

    def create_anonymous_token(self, session_hash: str, expires_in: int = 86400 * 30) -> str:
        """Issue an anonymous JWT session token containing zero personal identifiable information (PII)."""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": f"anon_{session_hash[:12]}",
            "session_hash": session_hash,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "compliance": "APP_1988_ZERO_PII",
        }

        encoded_header = self._b64url_encode(json.dumps(header).encode("utf-8"))
        encoded_payload = self._b64url_encode(json.dumps(payload).encode("utf-8"))

        signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
        signature = hashlib.sha256(signing_input + self.secret_key.encode("utf-8")).digest()
        encoded_signature = self._b64url_encode(signature)

        return f"{encoded_header}.{encoded_payload}.{encoded_signature}"

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify anonymous JWT signature and return token payload if valid."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            encoded_header, encoded_payload, encoded_signature = parts

            signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
            expected_sig = hashlib.sha256(signing_input + self.secret_key.encode("utf-8")).digest()

            if self._b64url_encode(expected_sig) != encoded_signature:
                return None

            payload = json.loads(self._b64url_decode(encoded_payload).decode("utf-8"))

            if payload.get("exp", 0) < time.time():
                return None  # Token expired

            return payload
        except Exception:
            return None
