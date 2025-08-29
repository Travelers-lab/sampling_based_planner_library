from setuptools import setup, find_packages

setup(
    name="sampling_planners",
    version="0.1.0",
    description="A Python library for sampling-based path planning with cost maps.",
    author="Chengjin Wang",
    author_email="2210991@tongji.edu.cn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.5.0",
    ],
    python_requires=">=3.7",
    license="MIT",
)