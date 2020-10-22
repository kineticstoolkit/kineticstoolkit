import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read()


setuptools.setup(
    name='kineticstoolkit',
    version=version,
    description='An open-source python library to facilitate research in biomechanics.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://felixchenier.uqam.ca/kineticstoolkit',
    author='Félix Chénier',
    author_email='chenier.felix@uqam.ca',
    license='Apache',
	license_files=['LICENSE.txt', 'NOTICE.txt'],
    packages=setuptools.find_packages(),
    install_requires=['pandas',
                      'pyqt5',
                      'scipy',
                      'matplotlib',
                      'scikit-learn',
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
    python_requires='>=3.8',
)
