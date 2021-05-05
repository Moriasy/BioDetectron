import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biodetectron",
    version="0.6.0",
    author="David Bunk",
    author_email="bunk@bio.lmu.de",
    description="Machine learning framework for biological image analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CALM-LMU/BioDetectron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "detectron2@https://github.com/facebookresearch/detectron2",
        "pandas",
        "scikit-image",
        "imgaug",
        "numpy",
        "pycocotools",
        "ffmpeg-python"
    ]
)