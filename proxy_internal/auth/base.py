"""Claude MITM auth cache (headers + body template captured by interceptor)."""

from typing import Optional

import aiohttp


class AuthCache:
    def __init__(self):
        self.headers: Optional[dict] = None
        self.body_template: Optional[dict] = None
        self.session: Optional[aiohttp.ClientSession] = None

    @property
    def is_ready(self):
        return self.headers is not None and self.body_template is not None
