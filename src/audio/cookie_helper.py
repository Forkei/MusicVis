"""One-time cookie setup: opens Chrome to YouTube for sign-in, exports cookies."""

import asyncio
import json
import os
import subprocess
import tempfile
import time

COOKIE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cookies.txt")
_CHROME_PATHS = [
    # Linux
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/snap/bin/chromium",
    # Windows
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    # macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def cookies_exist() -> bool:
    return os.path.exists(COOKIE_FILE) and os.path.getsize(COOKIE_FILE) > 100


def _find_chrome() -> str | None:
    for p in _CHROME_PATHS:
        if os.path.exists(p):
            return p
    return None


def open_login_browser() -> subprocess.Popen | None:
    """Open a Chrome window to YouTube for the user to sign in.
    Returns the Popen object (or None if Chrome not found).
    """
    chrome = _find_chrome()
    if not chrome:
        return None

    tmp_profile = tempfile.mkdtemp(prefix="musicvis_yt_")

    proc = subprocess.Popen(
        [
            chrome,
            f"--user-data-dir={tmp_profile}",
            "--no-first-run",
            "--remote-debugging-port=9225",
            "https://accounts.google.com/ServiceLogin?continue=https://www.youtube.com/",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc._tmp_profile = tmp_profile
    return proc


def export_cookies(proc: subprocess.Popen) -> bool:
    """Extract cookies from the running Chrome via CDP and save to cookies.txt.
    Returns True if YouTube cookies were found.
    """
    try:
        import websockets
    except ImportError:
        subprocess.run(
            ["pip", "install", "websockets"],
            capture_output=True,
        )
        import websockets

    return asyncio.run(_export_cookies_async())


async def _export_cookies_async() -> bool:
    try:
        import websockets
    except ImportError:
        return False

    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:9225/json/list", timeout=3)
        pages = json.loads(resp.read())

        # Find a YouTube page
        page_ws = None
        for p in pages:
            url = p.get("url", "")
            if "youtube.com" in url or "google.com" in url:
                page_ws = p.get("webSocketDebuggerUrl")
                break

        if not page_ws:
            # Fallback to browser endpoint
            resp2 = urllib.request.urlopen("http://localhost:9225/json/version", timeout=3)
            info = json.loads(resp2.read())
            page_ws = info.get("webSocketDebuggerUrl")

        if not page_ws:
            return False

        async with websockets.connect(page_ws, max_size=10**7) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Network.enable"}))
            await asyncio.wait_for(ws.recv(), timeout=3)

            await ws.send(json.dumps({"id": 2, "method": "Network.getAllCookies"}))

            for _ in range(20):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if msg.get("id") == 2:
                    cookies = msg.get("result", {}).get("cookies", [])
                    yt_cookies = [
                        c for c in cookies
                        if "youtube" in c.get("domain", "")
                        or "google" in c.get("domain", "")
                    ]

                    if not yt_cookies:
                        return False

                    # Check if actually logged in (look for SID or LOGIN_INFO)
                    cookie_names = {c["name"] for c in yt_cookies}
                    logged_in = "SID" in cookie_names or "LOGIN_INFO" in cookie_names or "__Secure-3PSID" in cookie_names

                    with open(COOKIE_FILE, "w") as f:
                        f.write("# Netscape HTTP Cookie File\n")
                        for c in yt_cookies:
                            domain = c["domain"]
                            dot = "TRUE" if domain.startswith(".") else "FALSE"
                            path = c.get("path", "/")
                            secure = "TRUE" if c.get("secure") else "FALSE"
                            expires = c.get("expires", 0)
                            if expires < 0:
                                expires = 0
                            name = c["name"]
                            value = c["value"]
                            f.write(f"{domain}\t{dot}\t{path}\t{secure}\t{int(expires)}\t{name}\t{value}\n")

                    return logged_in

    except Exception:
        return False
