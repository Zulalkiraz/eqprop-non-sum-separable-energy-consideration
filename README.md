# eqprop-interconnected-energy-function-consideration

# Equilibrium Propagation with Non-Sum-Separable Energy Consideration

This repository contains code for implementing and analyzing the Equilibrium Propagation (EqProp) algorithm with non-sum-separable energy considerations in analog neural networks. The project explores both traditional methods and a new power computation method for training neural networks on tasks such as the XOR problem.

## Repository Structure

- `eqprop_module_xor_pcm_vdm.py`: Contains necessary functions and modules for implementing the EqProp algorithm, including power dissipation calculations and voltage drop methods.
- `xor_pcm_vdm_eqprop.py`: Implements the XOR task using both the traditional EqProp method and the new power computation method. This script sets up the neural network, performs the training, and evaluates the performance.
- `starting_cond_analysis_pcm_vdm_eqprop.py`: Analyzes the starting conductance conditions for both the traditional method and the new power computation method, providing insights into the initial setup and its impact on learning.

## Getting Started

### Prerequisites

- Python 3.x
- `numpy` library
- `matplotlib` library
- `PySpice` library
- `ngspice` installed on your system

### Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:Zulalkiraz/eqprop-non-sum-separable-energy-consideration.git
    cd eqprop-non-sum-separable-energy-consideration
    ```

2. Install the required Python packages:

    ```bash
    pip3 install numpy matplotlib PySpice
    ```

### Running the Code

1. **Run the XOR Task**:

    This script sets up the neural network, performs the training using both the traditional EqProp method and the power computation method, and evaluates the performance on the XOR task.

    ```bash
    python3 xor_pcm_vdm_eqprop.py
    ```

2. **Run the Starting Conductance Analysis**:

    This script analyzes the initial conductance conditions and their impact on learning using both the traditional and power computation methods.

    ```bash
    python3 starting_cond_analysis_pcm_vdm_eqprop.py
    ```

## Project Overview


### Scripts Explanation

- **eqprop_module_xor_pcm_vdm.py**:
  - Contains functions for initializing circuits, calculating power dissipation, updating conductances, and performing the free and nudge phases of EqProp.

- **xor_pcm_vdm_eqprop.py**:
  - Implements the XOR task using both traditional and power computation methods, setting up the network, performing training, and evaluating results.

- **starting_cond_analysis_pcm_vdm_eqprop.py**:
  - Analyzes the starting conductance conditions, providing insights into how initial setups affect the learning process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

Zülal Kiraz: fatma.kiraz@telecom-paris and kirazulal@gmail.com
