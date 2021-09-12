import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

setuptools.setup(
    name="pymerlin-EMILLJUNGBERG",  # Replace with your own username
    version="0.1",
    author="Emil Ljungberg",
    author_email="ljungberg.emil@gmail.com",
    description="Python code for MERLIN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    # "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./"),
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'pymerlin=pymerlin.main_pymerlin:main',
            'h5viewer=pymerlin.main_tools:h5viewer',
            'nii2h5=pymerlin.main_tools:nii2h5',
            'h52nii=pymerlin.main_tools:h52nii'
        ]
    }
)
