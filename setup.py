import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepmath",
    version="0.0.2",
    author="Raiymbek",
    author_email="artmath1998@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mathraim/deepmath",
    download_url = "https://github.com/mathraim/deepmath/archive/0.0.2.tar.gz",
    packages=setuptools.find_packages(),
    install_requires = ['numpy','deepmath'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
