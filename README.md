# TVM's Relax Deployment Example in C++

This repository provides a minimal example for deploying Apache TVM's Relax IR using the C++ API.

> **Note:** This project only supports **[the latest TVM FFI](https://github.com/apache/tvm/pull/17920)**.

## Overview

This project demonstrates how to build and deploy a Relax IR model using TVM's C++ API. It includes instructions for setting up dependencies, building TVM from source, and running the example binary.

## Prerequisites

Ensure the following tools and libraries are installed on your system:

- **CMake**
- **Ninja** (build system)
- **Python** (for virtual environment setup)
- **LLVM** (with `llvm-config` available in your `PATH`)

## Building TVM from Source

Follow these steps to build TVM from source:

1. **Set up a Python virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required Python packages**:
   ```bash
   pip install cmake ninja setuptools cython
   ```

3. **Build TVM**:
   Navigate to the `3rdparty` directory and build TVM:
   ```bash
   cd 3rdparty
   ./build-tvm.sh --clean --llvm llvm-config
   ```

## Building the Example Project

Once TVM is built, follow these steps to build the example project:

1. **Configure the build using CMake**:
   ```bash
   cmake -B build -G Ninja
   ```

2. **Build the project using Ninja**:
   ```bash
   ninja -C build
   ```

## Exporting the Relax Library

Before running the example, export the Relax library using the provided Python script:

```bash
python export_relax_library.py
```

## Running the Example

After building the project, run the example binary:

```bash
./build/main
```

You'll see the output below:

```
Module(type_key= relax.VMExecutable)
Found vm_load_executable()
Found vm_initialization()
vm initialized
Found main()
Input array initialized
output: 
  1
  2
  3
  4
  5
  6
  7
  8
  9
```
