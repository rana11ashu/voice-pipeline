from setuptools import setup, find_packages

setup(
    name="qwen_megakernel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch"],
    package_data={"qwen_megakernel": ["../csrc/*.cu", "../csrc/*.cpp"]},
)
