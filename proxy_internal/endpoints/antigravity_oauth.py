"""Antigravity OAuth login + account management endpoints."""
import base64
import hashlib
import html
import logging
import secrets
import time
from urllib.parse import unquote

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
logger = logging.getLogger("proxy")


@router.get("/config/antigravity-login")
async def antigravity_login(project_id: str = ""):
    """Initiate Antigravity OAuth login flow."""
    import proxy

    # Start callback server if not running
    if proxy._oauth_callback_server is None:
        try:
            await proxy.start_oauth_callback_server()
        except Exception as e:
            logger.error(f"Failed to start OAuth callback server: {e}")
            return HTMLResponse(content=f"<h1>Failed to start OAuth callback server: {html.escape(str(e))}</h1>", status_code=500)

    # Generate PKCE verifier and challenge
    verifier = secrets.token_urlsafe(96)
    verifier_bytes = verifier.encode('utf-8')
    challenge_bytes = hashlib.sha256(verifier_bytes).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')

    # Store state
    state = secrets.token_urlsafe(16)
    proxy._cleanup_expired_oauth_states()
    proxy._oauth_states[state] = {
        "verifier": verifier,
        "project_id": project_id,
        "created_at": time.time(),
    }

    # Build authorization URL
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": proxy.ANTIGRAVITY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": proxy.ANTIGRAVITY_REDIRECT_URI,
        "scope": " ".join(proxy.ANTIGRAVITY_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    url_with_params = auth_url + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html><head><title>Antigravity Login</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:600px; margin:60px auto; padding:20px; text-align:center }}
  h1 {{ color:#f8fafc; font-size:1.5rem }}
  .status {{ color:#4ade80; font-size:0.9rem; margin:10px 0 }}
  .btn {{ display:inline-block; background:#4285f4; color:white; padding:14px 28px; border-radius:6px; text-decoration:none; font-weight:600; margin:20px 0 }}
  .btn:hover {{ background:#3367d6 }}
  .note {{ color:#94a3b8; font-size:0.85rem; margin-top:20px }}
</style></head>
<body>
  <h1>🔐 Antigravity Login</h1>
  <div class="status">✓ OAuth callback server ready on port {proxy.ANTIGRAVITY_OAUTH_PORT}</div>
  <p>Click below to sign in with your Google account</p>
  <a href="{url_with_params}" class="btn">Sign in with Google</a>
  <p class="note">After signing in, you'll be redirected back to complete setup.<br>
  Requires a US-associated Google account.</p>
</body></html>
""")


@router.delete("/config/antigravity-account/{email}")
async def remove_antigravity_account(email: str):
    """Remove an Antigravity account by email."""
    import proxy

    decoded_email = unquote(email)

    if proxy.antigravity_auth.remove_account(decoded_email):
        proxy._save_antigravity_auth()
        logger.info(f"Removed Antigravity account: {decoded_email}")
        return {"status": "removed", "email": decoded_email}
    return {"status": "not_found", "email": decoded_email}
