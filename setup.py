from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ("numpy", "torch")

testing_requires = ("pyimagetest", "pillow", "torchvision")

classifiers = (
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
)

setup(
    name="torchimagefilter",
    description="Image filtering in PyTorch",
    version="0.1",
    url="https://github.com/pmeier/torchimagefilter",
    license="GPLv3",
    author="Philip Meier",
    author_email="github.pmeier@posteo.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("test",)),
    install_requires=install_requires,
    extra_requires={"testing": testing_requires,},
    python_requires=">=3.6",
    classifiers=classifiers,
)
