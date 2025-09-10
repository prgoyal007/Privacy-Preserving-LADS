# Disclaimer
Although we utilize Biased ZipZip and Threshold ZipZip Trees in our experimental setup and results, we cannot disclose the code to the public. We apologize for any inconvenience. Please refer to the section where we show you how you can mimic the same Biased property with skip lists. 

# Privacy-Preserving Learning-Augmented Data Structures
This repository contains the experimental code, data structures, benchmarks, and figures accompanying the paper:

**Privacy-Preserving Learning-Augmented Data Structures**
*Authors: Goyal, et al.*


This repo provides:
- Source code for data structures and experiments described in the paper (except ZipZip Tree variants)
- Workload benchmarks (Zipfian, Inverse Power)

# How to obtain Biased property for any data structure
To Do...

# Setup
1. Make sure Python 3.10+ is installed
2. Install required Python packages from project root directory:
```bash
pip install numpy nltk matplotlib
```
3. Add `__init__.py` files (can be empty) to both subdirectories to mark them as packages:

```bash
tests/__init__.py
structures/__init__.py
```
This allows Python to recognize them as modules and handle imports correctly. 

4. Create the results directory structure from project root directory (required for tests to run correctly):
```bash
mkdir -p results/InversePowerTest
mkdir -p results/ROFZipfianTest
mkdir -p results/ROTZipfianTest
mkdir -p results/SizeTest
```
These directories will be ignored by Git (as specified in `.gitignore`) because the test scripts save data there. The tests will fail if these directories do not exist. 

## Running Tests
All test scripts are located in the `tests/` directory. To run a test:
1. Open a terminal and navigate to the project root directory:
```bash
C:\Privacy-Preserving-LADS>
```
2. Run the desired test using the `-m` flag:
```bash
python -m tests.{fileName}
```
3. Replace `{fileName}` with the test script name without the `.py` extension.
- Example:
```bash
python -m tests.ROF_ZipfianTest
python -m tests.ROT_ZipfianTest
python -m tests.InversePowerTest
python -m tests.SizeTest
```

## Why -m is used

The `-m` flag tells Python to run the script as a module. This ensures:
- Imports like `from structures.StaticRSL import *` or `from tests.DataGenerator import *` work correctly.
- Python resolves imports relative to the project root rather than the script’s folder.
