# BarebonesML: **ML From Scratch in Python**

This repository contains Python scripts that implement machine learning algorithms from scratch. The purpose of this repository is to provide simple and pedagogic implementations of machine learning algorithms to aid in understanding their inner workings. The focus is on clarity and simplicity, and the code is not intended to replace standard machine learning libraries in production use.

## Overview

- **Pedagogic Focus:** The primary goal of this repository is to serve as an educational resource. The code is intentionally kept simple to help individuals understand the fundamental concepts of various machine learning algorithms.

- **No Complicated Code:** You won't find complex implementations here. Instead, the scripts aim to be straightforward and easy to follow. This makes them suitable for those learning machine learning for the first time.

- **Not a Replacement for Libraries:** It's important to note that the code provided here is not meant to replace the functionality or efficiency of established machine learning libraries like scikit-learn or TensorFlow. These implementations are simplified versions for learning purposes.

## Contents

The repository is organized into folders, each dedicated to a specific machine learning algorithm. Each folder typically includes:

- A Python script implementing the algorithm from scratch.
- The details of the logic used and docs can be found [here](https://shrisharaob.github.io/tutorials/BareBonesML/index.html)
- Each folder contains a `demos.ipynb` notebook file that showcases the application of the algos on toy datasets


## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/shrisharaob/BarebonesML.git
    cd BarebonesML 
    ```

2. Explore the algorithm folders and select the one you're interested in.

3. Open the Python script to study the implementation. Comments are included to explain each step.

4. Experiment and Learn! Feel free to modify the scripts or use them as a starting point for your own implementations.

## Run on Docker

```bash
docker run -v <path_to_repository>/BarebonesML:/app -t bbml_app <path_to_python_script_to_run>
```

For example to run the file `linear_regiression.py` in the `supervised` folder, run the following: 
```bash
docker run -v <path_to_repository>/BarebonesML:/app -t bbml_app ./supervised/linear_regression.py 
```

## Contribution Guidelines

Contributions to enhance existing implementations or add new algorithms are welcome! Please follow the contribution guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

These implementations are meant for educational purposes and are not intended for production use. When working on real-world projects, consider using established machine learning libraries for efficiency, reliability, and additional features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
