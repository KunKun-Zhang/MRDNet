# Direction of Arrival (DOA) Models and Dataset

This repository provides the implementation and dataset generation scripts for Direction of Arrival (DOA) models, as detailed in Chapter 3 and Chapter 4 of the associated academic thesis. The DOA models are designed to perform accurately in various noisy and reverberant environments, facilitating robust sound source localization.

## Table of Contents

- [DOA Models in Noisy Environments](#doa-models-in-noisy-environments)
- [DOA Dataset Generation](#doa-dataset-generation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## DOA Models in Noisy Environments

### Overview

The DOA models implemented in this repository are tailored to operate under different noisy conditions, as outlined in Chapter 3 of the thesis. These models leverage various microphone array configurations and environmental settings to ensure high accuracy in sound source localization tasks.

### Directory Structure

All DOA model-related code is organized within the `doa/noise` directory. The models are categorized into subdirectories based on the specific environmental conditions and array configurations under which they are trained and validated:

1. **`diff_mic_num_normalize`**
   
   - **Description**: This subdirectory contains training and validation scripts for microphone arrays with varying numbers of microphones but normalized configurations.
   - **Contents**:
     - **Training Scripts**: Implementations for different array sizes.
     - **`train_for_variance` Folder**: Stores results from multiple experimental runs. Each run's outcome is documented as comments within the respective `net` files, facilitating variance analysis across different configurations.

2. **`fix_array`**
   
   - **Description**: Focuses on models utilizing fixed array configurations with subarray interleaving techniques.
   - **Contents**:
     - **Interleaving Scripts**: Code for interleaving subarrays to enhance model robustness.
     - **Model Variations**: Differences among models are annotated at the beginning of each `net` file, providing clarity on specific architectural or parameter changes.

3. **`same_size_diff_num`**
   
   - **Description**: Contains scripts for microphone arrays that maintain identical aperture sizes but vary in the number of microphones.
   - **Contents**:
     - **Training and Validation**: Scripts designed to evaluate the impact of microphone count on DOA estimation accuracy while keeping aperture size constant.

4. **`wind_noise`**
   
   - **Description**: Dedicated to models trained under wind noise conditions, simulating real-world outdoor environments.
   - **Contents**:
     - **Clipping Scenarios**: Subdirectories with "clip" in their names represent scenarios where microphone clipping occurs due to high-intensity wind noise. Other subdirectories handle unclipped conditions.
     - **Training Scripts**: Adapted to handle the dynamics introduced by wind noise, ensuring models can generalize across different noise intensities.

## DOA Dataset Generation

### Overview

The dataset generation scripts are essential for creating diverse and comprehensive datasets that simulate various acoustic environments and microphone array configurations. These datasets are crucial for training and evaluating the robustness of DOA models.

### Directory Structure

All dataset generation scripts are located within the `dataset/doa_data` directory, which is further organized into the following subdirectories:

1. **`car_dataset`**
   
   - **Description**: Generates datasets based on the SensIT dataset, tailored to simulate automotive acoustic environments.
   - **Contents**:
     - **Data Generation Scripts**: Tools for synthesizing sound sources and noise profiles typical in vehicular settings, facilitating the training of models for in-car applications.

2. **`different_mic_num`**
   
   - **Description**: Creates datasets for microphone arrays that have the same spacing between microphones but vary in the total number of microphones.
   - **Contents**:
     - **Array Configuration Scripts**: Enable the generation of data for arrays with different microphone counts, assessing the impact of array size on DOA estimation performance.

3. **`different_rt60`**
   
   - **Description**: Produces datasets with varying reverberation times (RT60), simulating different levels of reverberance in environments.
   - **Contents**:
     - **Reverberation Scripts**: Adjust RT60 parameters to create datasets that help models learn to operate effectively in both highly reverberant and minimally reverberant settings.

4. **`same_allSize`**
   
   - **Description**: Generates datasets for microphone arrays that maintain identical aperture sizes but have different spacing between microphones.
   - **Contents**:
     - **Spacing Configuration Scripts**: Facilitate the creation of datasets that explore how microphone spacing influences DOA estimation accuracy while keeping the overall array size constant.

## Repository Structure

```
├── diff_mic_num_normalize
├── fix_array
├── same_size_diff_num
├── wind_noise
├── dataset
│   └── doa_data
│       ├── car_dataset
│       ├── different_mic_num
│       ├── different_rt60
│       └── same_allSize
└── README.md
```

## Usage

To utilize the DOA models and generate the corresponding datasets, follow the instructions below:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/DOA-Models.git
   cd DOA-Models
   ```

2. **Set Up the Environment**

   Ensure that you have Python installed along with the necessary dependencies. It is recommended to use a virtual environment.

   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Generate Datasets**

   Navigate to the desired dataset generation subdirectory and execute the generation scripts.

   ```bash
   cd dataset/doa_data/car_dataset
   python generate_car_dataset.py
   ```

4. **Train DOA Models**

   Access the specific model subdirectory and run the training scripts.

   ```bash
   cd ../../doa/noise/diff_mic_num_normalize
   python train_model.py
   ```

5. **Evaluate Models**

   Use the provided validation scripts to assess model performance.

   ```bash
   python validate_model.py
   ```

## Acknowledgements

This work builds upon the SensIT dataset and utilizes various open-source libraries for audio processing and machine learning. Special thanks to the contributors and maintainers of these resources.

## License

This project is licensed under the [MIT License](LICENSE).
