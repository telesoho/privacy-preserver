import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="privacy_preserver",  # Replace with your own username
    version="0.0.1",
    author="telesoho",
    author_email="telesoho@gmail.com",
    description="Anonymizing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/telesoho/privacy-preserver",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.3.2',
        'numpy>=1.21.2',
        'kmodes'
    ]
)