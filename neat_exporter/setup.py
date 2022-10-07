from setuptools import setup

setup(
    name='neat-python-exporter',
    version='0.1',
    author='Kevin Hernandez',
    author_email='kevinh16.kh@gmail.com',
    maintainer='Kevin Hernandez',
    maintainer_email='kevinh16.kh@gmail.com',
    url='https://github.com/CodeReclaimers/neat-python',
    license="BSD",
    description='A JSON exporter of NEAT (NeuroEvolution of Augmenting Topologies) models created by the package neat-python',
    long_description='Python implementation of a tool to export NEAT (NeuroEvolution of Augmenting Topologies) models' +
                     ' into a general JSON format that can be used by other people and software. '
                     'Developed by Kevin Hernandez and inspired by the package neat-python.',
    long_description_content_type='text/x-rst',
    packages=['neat_export'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ]
)