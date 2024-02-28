import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="",
    version="1.0.0",
    author="Avinash Kumar Chaurasia, Matin Fallahi",
    author_email="avichaurasia@gmail.com",
    description="NeuroBench: An open-source benchmarking framework for standardization of methodolgy in brainwave authentication research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Avichaurasia/Brain-Models",
    license='GNU GPLv3+',
    package_dir={"": "brainModels"},
    packages=setuptools.find_packages(where="brainModels"),
    install_requires=[
        "numpy",
        "scipy", 
        "mne", 
        "pandas", 
        "scikit-learn", 
        "matplotlib", 
        "seaborn", 
        "pooch", 
        "requests" 
        "tqdm", 
        "zipfile36", 
        "statsmodels",    
        "mat73", 
        "tensorflow", 
        "tensorflow_addons"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
