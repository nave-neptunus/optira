"""Optional ngrok tunnel. Reads the auth token from the environment only.

The original notebook ran the backend on a Colab VM behind an ngrok tunnel.
Set NGROK_AUTHTOKEN in your environment (see .env.example). There is no
authentication on the API, so a public tunnel exposes it to anyone with the URL.
"""

from __future__ import annotations

import os
import sys


def start_tunnel(port: int = 5000) -> str:
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        sys.exit("NGROK_AUTHTOKEN is not set. Export it in your shell, then retry.")
    from pyngrok import ngrok  # lazy, optional (pip install pyngrok)

    ngrok.set_auth_token(token)
    url = str(ngrok.connect(port))
    print(f"Tunnel open: {url}/analyze")
    return url
