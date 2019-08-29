"""
allow tdub command line interface to be called from ``python -m tdub``
"""

from ._app import cli

if __name__ == "__main__":
    cli()
