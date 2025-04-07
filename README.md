# TRY_TO_CALCULATE_CHIRP_MASS_OUT_OF_LIGO_STRAINS

This repository contains scripts and resources to estimate black hole masses from LIGO strain data using a novel approach based on segmented spacetime. The model combines both a segmented FFT analysis and refined matched filtering to produce realistic mass estimates.

## Contents

- **mass-calculation-final-try.py**  
  The main Python script implementing both methods (segmented FFT analysis and matched filtering) for mass estimation.

- **mass-calculation-try.pdf**  
  A PDF paper that details the theoretical background and methodology of the segmented spacetime approach.

- **dependencies-install-2.sh**  
  A shell script to create and set up the required Conda environment (`ligo_env`) with all necessary dependencies (e.g., PyCBC, LALSuite).

- **README.md**  
  This file.

## System Requirements

- Debian Linux (or a compatible system)
- Conda (Anaconda or Miniconda)
- Bash (for executing shell scripts)
- Python 3.10 (as specified in the Conda environment)

## Setup

1. **Make the shell script executable:**  
   Open a terminal in the repository directory and run:
   
chmod a+x dependencies-install-2.sh


2. **Install dependencies:**  
Run the installation script:

./dependencies-install.sh

This script creates the Conda environment `ligo_env` and installs PyCBC, LALSuite, and other required packages.

3. **Activate the Conda environment:**  

conda activate ligo_env


4. **Run the analysis script:**  
Execute the script using:

python mass-calculation-try.py

**Note:** Although the file extension is `.pdf`, it contains the complete Python code. Please ensure you are using the correct version.

## How It Works

The main script performs the following steps:

- **Data Loading & Preprocessing:**  
The LIGO strain data (in a `.txt` file with 3 header lines) is loaded, high-pass filtered (cutoff at 50 Hz), DC-offset corrected, and normalized.

- **Frequency Analysis:**  
Multiple techniques are combined for improved frequency estimation:
- **Zero-Padding and Optimal Windowing:** Uses a Hann window and optional zero-padding to increase frequency resolution.
- **Welch’s Method:** Computes the power spectral density (PSD) using overlapping segments for a robust estimate.
- **Sub-Peak Interpolation:** Refines the dominant peak frequency using parabolic interpolation.
- **Wavelet Analysis (optional):** Performs a continuous wavelet transform (if PyWavelets is installed) for an alternative frequency estimate.

- **Segmentation & Dynamic Calibration:**  
The signal is divided into overlapping segments. For each segment, an effective frequency is computed using a dynamic calibration factor derived from the segment’s amplitude statistics.

- **Matched Filtering:**  
A coarse search is performed over a mass range from 5 to 100 M☉, followed by a fine-tuned search around the best value, using IMRPhenomD waveforms.

- **Results & Saving:**  
The script outputs the estimated black hole masses from both methods and the combined chirp mass, then saves the results to a file.

## Further Information

For detailed explanations of the theoretical background and methodology, please refer to:
**mass-calculation-try.pdf**

## License

*(Insert appropriate license information here, if applicable.)*

## Contact

For questions or further information, please contact [mail[at]error.wtf].



