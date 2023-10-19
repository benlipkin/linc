from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

setup(
    description="Evaluation framework for LINC.",
    long_description=readme,
    license="MIT",
    packages=["eval"],
    install_requires=requirements,
    python_requires=">=3.10",
)
