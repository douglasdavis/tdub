from setuptools import setup
from setuptools import find_packages
import pathlib


project_root = pathlib.PosixPath(__file__).parent

with (project_root / "requirements.txt").open("r") as f:
    requirements = f.read().splitlines()

with (project_root / "README.md").open("r") as f:
    readme = f.read()

with (project_root / "tdub" / "__init__.py").open("r") as f:
    for line in f.readlines():
        if "__version__ = " in line:
            version = line.strip().split(" = ")[-1][1:-1]

with (project_root / "docs" / "requirements.txt").open("r") as f:
    dev_requirements = f.read().splitlines()

setup(
    name="tdub",
    version=version,
    scripts=[],
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["tdub = tdub._app:cli"]},
    description="tW analysis tools",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Doug Davis",
    author_email="ddavis@ddavis.io",
    maintainer="Doug Davis",
    maintainer_email="ddavis@ddavis.io",
    license="BSD 3-clause",
    url="https://github.com/douglasdavis/tdub",
    test_suite="tests",
    python_requires=">=3.7",
    install_requires=requirements,
    tests_require=["pytest>=5.2"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
