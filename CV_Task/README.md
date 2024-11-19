# Task 2: Computer Vision - Sentinel-2 Image Matching

This task focuses on developing an algorithm or model for matching features in satellite images. The goal was to create a robust system that can detect and match key points across images of Sentinel-2 data, specifically to monitor deforestation in Ukraine.

## Libraries Used

The following libraries were used for dataset processing, image display, and applying algorithms to create and match key points:

- **OpenCV**: For image processing and feature extraction.
- **rasterio**: For reading and writing geospatial raster data.
- **matplotlib**: For image visualization.
- **geopandas**: For geospatial data manipulation.

## Dataset

The dataset used for this task is the "Deforestation in Ukraine from Sentinel-2 data by Quantum" dataset, available on Kaggle:

- [Deforestation in Ukraine dataset on Kaggle](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine)

## Algorithm Used

For the image comparison algorithm, the **SIFT (Scale-Invariant Feature Transform)** method was used to create descriptors for key points. It is known for its accuracy, which made it ideal for this task. **ORB (Oriented FAST and Rotated BRIEF)** was also considered, but it was less accurate compared to SIFT, despite being faster. After extracting features, the **BFMatcher (Brute Force Matcher)** was applied to match the key points across images.

After applying these machine learning methods, one of the following matches was obtained:

![Image Match Example](https://github.com/oleh17v/Quantum-Internship-Test-Task/blob/main/CV_Task/res_001.png)

## Environment Setup

### 1. Install Dependencies

If you're using **Google Colab**, the required dependencies are already pre-installed for the most part. However, you may need to install certain libraries, such as OpenCV or rasterio. You can install them by running the following commands in a code cell in the notebook:

```bash
pip install -r requirements.txt
```
### 2. Using Google Colab

If you are using **Google Colab**, you don't need to worry about the environment setup or installing the kernel. Just open the notebook and connect to the Colab runtime. But don't forget to have image_matching_algorithm.py script in your directory.

### 3. Install Jupyter Kernel (for Local Development)
```bash
pip install ipykernel
python -m ipykernel install --user --name <your_env_name> --display-name "Python (<your_env_name>)"
```

### 4. Launch Jupyter Notebook (for Local Development)

```bash
jupyter notebook
```
