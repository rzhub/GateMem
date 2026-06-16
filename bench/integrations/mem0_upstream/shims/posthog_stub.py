"""No-op stub for `posthog`.

Vendored Mem0 upstream imports `posthog` for telemetry. Benchmark runs should be
able to proceed without requiring that dependency.

We inject this module into `sys.modules` only when Mem0 upstream backend is used.
"""

from __future__ import annotations


class Posthog:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        pass

    def capture(self, *args, **kwargs):
        return None

    def identify(self, *args, **kwargs):
        return None


__all__ = ["Posthog"]
