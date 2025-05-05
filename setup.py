from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f.readlines() if not line.startswith("#")]

setup(
    name="sequencer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    author="Mahmoud Badawy",
    author_email="mahmoud1816yasser@gmail.com",
    description="GUI sequencer for controlling the hardware of the lab using the ADwin system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mahmoud-Yasser-18/BEC-Sequencer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)