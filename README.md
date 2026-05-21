# Roman Vision: CNN

An CNN pipeline designed to process multi-year photometric data for the detection and characterization of long-period exoplanets, inspired by the operational requirements of the Nancy Grace Roman Space Telescope.

## Project Objective

The primary goal of "Roman Vision" is to push the boundaries of automated exoplanet vetting by transitioning from single-sector analysis to **Multi-Year Data Stitching**. Unlike standard pipelines that analyze short 27-day windows, this system stitches together up to 7 years of NASA TESS data, enabling the detection of true Earth analogs (long-period planets in the Habitable Zone).

It leverages a hybrid architecture, combining rigorous astrophysical algorithms with a Deep Convolutional Neural Network (CNN) for robust classification.

## Architecture & Methodology

The pipeline operates in three distinct phases:

1.  **Data Engineering (Multi-Sector Stitching):**
    * Fetches high-cadence SPOC data via the `lightkurve` API.
    * Implements an **Auto-Heal mechanism** to recover from NASA API limits and corrupt cache files dynamically.
    * Stitches multiple sectors and applies a flattening filter (window length = 901) to remove long-term stellar trends.

2.  **Astrophysical Vetting (Pre-screener):**
    * A physics-based algorithm calculates the Odd/Even transit depth ratio to filter out Eclipsing Binaries before they reach the neural network, reducing computational overhead and False Positives.

3.  **AI Classification (Deep CNN):**
    * Phase-folded lightcurves are processed by a trained TensorFlow CNN (`PlanetValidator`).
    * The model calculates the **Euclidean Distance** between the target's extracted features and an ideal `planet_centroid`, ensuring highly interpretable results compared to standard black-box classifiers.

## ⚙️ Installation

To run this pipeline, you need Python 3.9+ and a stable internet connection.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/roman-vision.git](https://github.com/yourusername/roman-vision.git)
    cd roman-vision
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have TensorFlow, lightkurve, astropy, and pandas installed).*

## How to Run

1.  **Prepare your targets:** Ensure your `data/interim/targets_to_scan.csv` is populated with the `TIC_ID` and `Period` of your targets.
2.  **Execute the Pipeline:**
    ```bash
    python main.py
    ```
3.  **Review Results:** The pipeline runs in a multithreaded environment. Real-time logging will display the vetting process, and the final results will be saved in `reports/scan_results_stitched.csv`.

## Hardware Requirements

* The pipeline features memory-optimized execution (In-Memory RAM Caching) but is I/O bound.
* Recommended: Multicore CPU, NVMe SSD, and a high-speed internet connection for API fetching.
