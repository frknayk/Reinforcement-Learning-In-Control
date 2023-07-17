import setuptools
from setuptools import setup

setup(
    name="gym_control",
    version="0.0.1",
    install_requires=["ray", "gym", "rllib"],
    description="Control systems with Reinforcement Learning and Dynamic Programming :rock:",
    author="frknayk",
    author_email="furkanayik@outlook.com",
    packages=setuptools.find_packages(),
)
