import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    installation_requirements = fr.readlines()

setuptools.setup(
    name="literegistry",
    version="1.0.2",
    author="Goncalo Faria",
    author_email="gfaria@cs.washington.edu",
    description="Package for implementing service discovery in a really lite way.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goncalorafaria/lightregistry",
    packages=setuptools.find_packages(),
    install_requires=installation_requirements,
    python_requires=">=3.6.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "literegistry = literegistry.cli:main",
        ],
    },
)
