import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trapallada",
    version="0.0.1",
    author="Manuel A. Vázquez",
    author_email="manuavazquez@gmail.com",
    description="Analyze light curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manuvazquez/trapallada",
    packages=setuptools.find_packages(),
    install_requires=['astropy', 'astroquery', 'pandas', 'jax', 'optax', 'equinox', 'jaxtyping', 'numpyro'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': ['trap_fit.py=trapallada.core:cl_fit'],
    }
)
