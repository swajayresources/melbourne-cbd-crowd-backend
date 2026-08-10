/* Zero-PII anonymous authentication (APP / Privacy Act 1988 compliant).
 *
 * Flow:
 *   1. Browser generates an ECDSA P-256 keypair with the WebCrypto API.
 *   2. The PUBLIC key (JWK) is POSTed to /api/v1/auth/session.
 *   3. Backend hashes it -> session_hash, returns an anonymous JWT.
 *   4. The PRIVATE key is kept only in this browser (localStorage) and is
 *      used to prove continuity on later visits. No name, email or device
 *      identifier ever leaves the device.
 */
(function () {
  "use strict";

  const KEY_STORAGE = "mpl_anon_key";       // ECDSA P-256 private key (JWK)
  const SESSION_STORAGE = "mpl_anon_session"; // { session_hash, token }

  /* ---------- helpers ---------- */
  function b64urlFromJwk(jwk) {
    const json = JSON.stringify({ kty: jwk.kty, crv: jwk.crv, x: jwk.x, y: jwk.y });
    return btoa(unescape(encodeURIComponent(json)))
      .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  async function generateKeyPair() {
    return crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      true,                                   // extractable so the key persists across visits
      ["sign", "verify"]
    );
  }

  async function exportPublicString(keyPair) {
    const jwk = await crypto.subtle.exportKey("jwk", keyPair.publicKey);
    return b64urlFromJwk(jwk);
  }

  async function importStoredKey(privateJwk) {
    return crypto.subtle.importKey(
      "jwk",
      privateJwk,
      { name: "ECDSA", namedCurve: "P-256" },
      true,
      ["sign"]
    );
  }

  async function createSession(publicKeyStr) {
    const r = await fetch("/api/v1/auth/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ public_key: publicKeyStr }),
    });
    if (!r.ok) throw new Error("session creation failed");
    return r.json();
  }

  /* ---------- session lifecycle ---------- */
  let current = null; // { token, session_hash, preferences }

  async function ensureSession() {
    if (current) return current;

    let keyPair = null;
    const storedKey = localStorage.getItem(KEY_STORAGE);
    if (storedKey) {
      try {
        keyPair = await importStoredKey(JSON.parse(storedKey));
      } catch (e) {
        localStorage.removeItem(KEY_STORAGE);
      }
    }

    if (!keyPair) {
      keyPair = await generateKeyPair();
      const privateJwk = await crypto.subtle.exportKey("jwk", keyPair.privateKey);
      localStorage.setItem(KEY_STORAGE, JSON.stringify(privateJwk));
    }

    const publicKeyStr = await exportPublicString(keyPair);
    const session = await createSession(publicKeyStr);
    localStorage.setItem(SESSION_STORAGE, JSON.stringify({
      session_hash: session.session_hash,
      token: session.token,
    }));

    current = session;
    renderStatus();
    return current;
  }

  function signOut() {
    localStorage.removeItem(KEY_STORAGE);
    localStorage.removeItem(SESSION_STORAGE);
    current = null;
    renderStatus();
  }

  /* ---------- preferences sync (personalisation, zero PII) ---------- */
  async function loadPrefs() {
    await ensureSession();
    const r = await fetch("/api/v1/user/prefs", {
      headers: { Authorization: "Bearer " + current.token },
    });
    if (!r.ok) throw new Error("prefs load failed");
    const data = await r.json();
    current.preferences = data.preferences || {};
    return current.preferences;
  }

  async function savePrefs(prefs) {
    await ensureSession();
    const r = await fetch("/api/v1/user/prefs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + current.token,
      },
      body: JSON.stringify({ preferences: prefs }),
    });
    if (!r.ok) throw new Error("prefs save failed");
    const data = await r.json();
    current.preferences = data.preferences || {};
    return current.preferences;
  }

  /* ---------- status chip in the prefs bar ---------- */
  function renderStatus() {
    const bar = document.querySelector(".prefs");
    let chip = document.getElementById("sessionStatus");
    if (!bar) return;
    if (!chip) {
      chip = document.createElement("span");
      chip.id = "sessionStatus";
      chip.className = "session-chip";
      bar.appendChild(chip);
    }
    if (current && current.session_hash) {
      const short = current.session_hash.slice(0, 8);
      chip.className = "session-chip session-chip-active";
      chip.title = "Anonymous session \u2014 no name, email or device data is stored. Click to sign out.";
      chip.textContent = "\uD83D\uDD12 Anonymous \u00B7 " + short;
      chip.onclick = (e) => {
        e.stopPropagation();
        if (confirm("Sign out of this anonymous session? Display preferences will stay on this device.")) {
          signOut();
        }
      };
    } else {
      chip.className = "session-chip";
      chip.title = "No active session";
      chip.textContent = "\uD83D\uDD13 No session";
      chip.onclick = (e) => {
        e.stopPropagation();
        ensureSession().catch(() => {
          chip.textContent = "\u26A0 Session unavailable";
        });
      };
    }
  }

  /* ---------- public API ---------- */
  window.AuthAPI = {
    ensureSession,
    signOut,
    loadPrefs,
    savePrefs,
    getToken: () => (current ? current.token : null),
    getSessionHash: () => (current ? current.session_hash : null),
  };

  /* Auto-start on load when WebCrypto is available (HTTPS/localhost only).
   * Failures are silent: the app works without a session. */
  document.addEventListener("DOMContentLoaded", () => {
    if (!window.crypto || !crypto.subtle) {
      const bar = document.querySelector(".prefs");
      if (bar) {
        const chip = document.createElement("span");
        chip.id = "sessionStatus";
        chip.className = "session-chip";
        chip.title = "WebCrypto unavailable (needs HTTPS or localhost)";
        chip.textContent = "\u26A0 Session unavailable";
        bar.appendChild(chip);
      }
      return;
    }
    ensureSession().catch(() => {});
  });
})();
