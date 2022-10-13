# neat-python-utility

Neat Python Utility is a library developed with the intention of integrating some
Quality of Life features related to the original [NEAT Python Library](https://github.com/CodeReclaimers/neat-python).

As the name of the library implies, there are multiple utilities such as:
- Easy generalized algorithm setup
- Easy and automated logs setup
- Integrated statistics reports
- Automated Network visualization
- Easy checkpoint training manipulation
- Export model in a JSON format

Think of this library as an extension or as a complement of the original one.

**I did not add anything new to the original Neat Python Library, I worked on adding
new miscellaneous features**

To install the library, execute the following command:

```
pip install git+https://github.com/KevinHern/neat-python-utility
```

Once it is done, check out [this example](https://github.com/KevinHern/neat-python-utility/tree/main/tests/usage_example) to learn how to use the functionalities!

# Quickstart

The big picture to make the library work as intended, have the following directory structure
for your project:

- **artificial_intelligence/**:
This directory will contain files associated with artificial intelligence stuff.
The most notable file is the NEAT configuration file that the original neat-python library
needs in order to set up the algorithm the way you want.
- **simulation/**:
This directory will contain all the files that you may need to run a simulation about
the problem you desire to solve using NEAT
- **main.py**: A simple file with few lines of code used to start the simulation
and the algorithm itself.

This library will enforce you to have such directory structure to create good
habits. Keeping everything organized and separated, will make it easier to debug.

## Notes

- The configuration file that contains parameters to set up properly the NEAT algorithm
**needs** to be named **config-feedforward.txt**
- The export feature **only** works with **feed_forward networks**