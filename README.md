# Exploring Data Efficiency of Foundation Models for Fine-Grained Image Classification of Lepidopteran Species

This is the work for a Bachelor Thesis (B.Sc.), absolved at the explainable AI Chair (xAI) of the Otto Friedrich University Bamberg.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Notes for Project Setup](#notes-for-project-setup)
- [Requirements](#requirements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project tests data efficiency in the domain of fine-grained image classification. Therefore, a dataset comprising 1.39 million images of butterflies and moths is split into varying label- and sample amounts, to test its impact on the performance of foundation models and classifiers trained on their extracted feature embeddings.

As the dataset is provided by GBIF (and Observation.org), it consists of citizen science data. To tackle the challenges of strongly varying image quality, this work tries to leverage the generalization capabilities of different foundation models, specifically DINOv2-Giant and ResNet-50 using ImageNet1k_V2 weights.

Trained on their embeddings were a Logistic Regression and K-Nearest Neighbors Classifier.

## Dataset

Links to the images used for feature extraction are given in the `dataset_top589_max3000_fex_statusupdate.csv` dataset. The original dataset (hosted on GBIF) can be accessed under the following [link](https://doi.org/10.15468/dl.8byj47). The images vary strongly in quality, some example images are given below.
<p align="center">
    <img src="data/example_samples/3725704126_27621411.jpg" alt="sample1" height="200px">
    <img src="data/example_samples/3726205023_30566083.jpg" alt="sample2" height="200px">
    <img src="data/example_samples/4881370136_87361092.jpg" alt="sample3" height="200px">
</p>

## Results

![Cross-Validation Accuracies](data/analysis_results/graphs/accuracies_cv_top_parameter.png)

Some accuracies derived using cross-validation for different dataset variants and foundation model - classifier combinations. Further results are given in the notebooks numbered from 07 to 11.

## Project Structure

- **data/**: Contains the dataset and processed data.
- **notebooks/**: Contains the Jupyter notebooks for analysis and creation of datasets, analysis of results, and further helpful tools.
- **scripts/**: Contains the scripts for feature extraction, testing, and evaluation.
- **src/**: Contains the source code for custom dataset classes and utility functions.

## Notes for Project Setup

Since the data used for this project is not available due to its size, it would have to be retrieved again by whoever wants to reproduce, improve, or investigate this work.

Links to the images are given as stated above, advised download speed is 2 images per second, the `01_image_downloader.py` can be used for this purpose.
After quality control, the feature embeddings of the two foundation models can be extracted using the `05_feature_extractor.py` script. From there on, tests provided in the scripts numbered from 06 to 09 can be used for experimenting.

Some data to start with is given in the `data` folder, detailed label-level confidence results or similar data is unfortunately not given due to GitHubs file-size limiatations.

## Requirements

- Python 3.8.10
- PyTorch
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- umap-learn

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks go to Jonas Alle (M.Sc.) who supervised this project and contributed greatly with valuable explanations, advice, experience, and good humor! Furthermore, I want to thank Prof. Dr. Ledig for being my first supervisor and letting me do the thesis at his chair.
