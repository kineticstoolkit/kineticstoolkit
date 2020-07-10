import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ktk',
    version='0.0.4',
    description='What will become a great library to analyze biomechanics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://felixchenier.uqam.ca',
    author='Félix Chénier',
    author_email='chenier.felix@uqam.ca',
    license='Apache',
    packages=setuptools.find_packages(),
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'matplotlib',
                      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.7',
)
