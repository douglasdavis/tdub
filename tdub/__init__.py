"""
tW analysis tools
"""

__version__ = "0.0.22"


def setup_logging():
    import logging

    # fmt: off
    logging.basicConfig(
        level=logging.INFO, format="{:22}  %(levelname)8s  %(message)s".format("[%(name)s : %(funcName)s]")
    )
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))
    # fmt: on
