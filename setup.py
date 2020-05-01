import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='kineticstoolkit',
    version='0.1a',
    description='What will become a great tool to analyze biomechanics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://felixchenier.com',
    author='Félix Chénier',
    author_email='chenier.felix@uqam.ca',
    license='Copyright',
    packages=setuptools.find_packages(),
    install_requires=['pandas',
                     ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL2 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.7',
)
