# Monte Carlo Simulation for VR-Throws

Python code to run the model by Trommershäuser et al. (2003) for VR throwing data.
The model estimates the optimal shift away from the penalty zone given (a) participants' endpoint variance, (b) penalty/rewards, and (c) the distance between the circles.

The script takes participants' throwing data (input data: 'data_exp1' from folder 'in')  
And generates for each participant and condition:

- Expected gain landscapes: Expected gain a function of potential horizontal shifts (output: excel files to folder 'out')
- Optimal horizontal shift: Shift that maximizes expected gain (output: adds a column 'optimal_aimpoint_x' to the file 'data_exp1' in folder 'out')

## Installation

Before running the scripts, ensure Python and the required libraries are installed on your system. Install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn
```

This will calculate everything on the cpu, if you need faster computation speed, this script is prepared to run on the gpu. To make use of this, make sure to have cuda installed. Follow those [instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). After this, you will have to install `cupy` which can act as a drop-in replacement of numpy, but will use cuda in the background.

```bash
pip install cupy-cuda12x
```

Keep in mind that the above command works in a environment where cuda version 12 is installed, this might differ for a different version.

## Scripts and How to Run Them

### `calculate_optimal_aimpoints.py`

Calculates optimal aimpoints based on distributions from actual trial data.

**Usage**:

```bash
python calculate_optimal_aimpoints.py [-i INPUTFILE] [-o OUTPUTFOLDER] [--use_vp_endpoint_variance] [--num_endpoints NUM_ENDPOINTS]
```

**Arguments**:

- `-i, --inputfile`: Path to the input Excel file. Default is 'in/data_exp1.xlsx'.
- `-o, --outputfolder`: Directory for all output files. Default is 'out/'.
- `--use_vp_endpoint_variance`: Utilize endpoint variance from the complete subject data for simulation. Default is `False`.
- `--num_endpoints`: Set the number of endpoints to simulate per aimpoint. Default is `1000000`.

The analyis corresponds to the one performed for the paper, when you call the script without setting parameters (i.e. when using defaults).

### monte_carlo_utils.py

Shared utilities and functions such as `monte_carlo()` for running simulations and `get_endpoints()` for determining suitable randomness. This file supports the functionality of the main scripts.
