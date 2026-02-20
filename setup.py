from setuptools import setup, find_packages

setup(
    name="loop-generator",
    version="0.1.0",
    description="AI-powered loop generator with Python GenAI integration and C++ DSP engine",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.1",
        "google-genai>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "loop-generator=cli.main:main",
        ],
    },
)
