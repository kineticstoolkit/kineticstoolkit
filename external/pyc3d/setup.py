import os
import setuptools

setuptools.setup(
    name='c3d',
    version='0.3.0',
    py_modules=['c3d'],
    author='UT Vision, Cognition, and Action Lab',
    author_email='leif@cs.utexas.edu',
    description='A library for manipulating C3D binary files',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.rst')).read(),
    license='MIT',
    url='http://github.com/EmbodiedCognition/py-c3d',
    keywords=('c3d motion-capture'),
    install_requires=['numpy'],
    scripts=['scripts/c3d{}'.format(s) for s in '-metadata -viewer 2csv 2npz'.split()],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        ],
    )
