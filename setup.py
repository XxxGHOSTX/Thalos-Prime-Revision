from setuptools import find_packages, setup

setup(
    name="thalos-prime",
    version="0.1.0",
    description="Autonomous Epistemic Compiler with SBI integration",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "cryptography>=42.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "httpx>=0.27.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "thalos-prime=thalos_prime.core:main",
        ]
    },
)
