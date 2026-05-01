"""Smoke tests for real third-party CLI integrations.

These tests run only in the path-gated CI smoke job (and locally when the
relevant credentials are present). They are excluded from the default
``pytest`` run by the ``-m 'unit or integration'`` selector in
``pyproject.toml`` and require the ``smoke`` marker to be requested
explicitly.
"""
