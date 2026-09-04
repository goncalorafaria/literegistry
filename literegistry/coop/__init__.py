"""Cross-process cooperation primitives used by LiteRegistry deployments.

Import concrete APIs from ``literegistry.coop.ports``, ``.artifacts``,
``.endpoints``, or ``.redis``. Keeping this package initializer side-effect free
also permits those modules to run cleanly through ``python -m``.
"""
