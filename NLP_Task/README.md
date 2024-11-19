# Task 1. Natural Language Processing. Named Entity Recognition (NER)

In this task, the goal was to train a model for the Named Entity Recognition (NER) task to identify mountain names in the text.

To accomplish this, we used the following libraries:
- **pandas**, **scikit-learn**: for data manipulation and applying metrics.
- **PyTorch**, **Transformers**: for training the neural network and downloading pre-trained models.

The dataset used for this task is a generated mountain entity recognition dataset.

For the method of solving the task, pre-trained models from the **Transformers** library, specifically the **BERT** neural network, were used for tokenization and classification simultaneously. The BERT model showed excellent performance with minimal loss and high accuracy.

### Confusion Matrix:
![Confusion Matrix](https://github.com/oleh17v/Quantum-Internship-Test-Task/blob/main/NLP_Task/results.png)

## Environment Setup

### 1. Install Dependencies

If you're using **Google Colab**, most dependencies are already pre-installed. However, you may need to install additional libraries, such as **OpenCV** or **rasterio**. You can do this by running the following command in a code cell:

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
