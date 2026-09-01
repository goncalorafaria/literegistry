import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    installation_requirements = fr.readlines()

bm25_requirements = [
    "pyserini==1.6.0; python_version < '3.12'",
    "pyserini==2.3.0; python_version >= '3.12'",
]

setuptools.setup(
    name="literegistry",
    version="1.0.40",
    author="Goncalo Faria",
    author_email="gfaria@cs.washington.edu",
    description="Package for implementing service discovery in a really lite way.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goncalorafaria/lightregistry",
    packages=setuptools.find_packages(),
    install_requires=installation_requirements,
    extras_require={
        "podman_client": ["literegistry-podman-client>=0.1.2"],
        "podman_beaker": ["literegistry-podman-beaker>=0.2.8"],
        "base_deployment": ["literegistry-base-deployment>=0.1.0"],
        "bm25": bm25_requirements,
        "all": [
            "literegistry-podman-client>=0.1.2",
            "literegistry-podman-beaker>=0.2.8",
            "literegistry-base-deployment>=0.1.0",
            *bm25_requirements,
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "literegistry = literegistry.cli:main",
        ],
    },
)
