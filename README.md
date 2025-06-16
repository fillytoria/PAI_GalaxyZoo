# Galaxy Zoo - Galaxy Classification Project

This project aims to classify galaxies based on their images and a set of questions describing the galaxy morphology. 
It is based on the Kaggle Galaxy Zoo Challenge, and prepared as a project for the course "Physics Applications of AI" at Université de Genève during the summer term 2025. 

## Task Overview

#### Exercise 1: Galaxy Classification 

The first exercise aims to classify galaxies into three types (`Class1.1`, `Class1.2`, `Class1.3`) using a Convolutional Neural Network (CNN). 
To decide whether the galaxy is smooth and round (Class1.1), has a feature or disk (Class 1.2) or is a flawed image (Class 1.3), the first three labels are used, and tranformed into one-hot encoded vectors. 
The dataset is preprocessed, filtered, and split into training and validation sets. The model is trained to predict galaxy types based on image data, and various diagnostics such as ROC curves, loss/accuracy curves, and classification reports are generated to evaluate the model's performance.

- **Data Preprocessing**:
  - Crop and resize images to 64x64 pixels
  - Filters images based on label thresholds
  - Saves filtered labels and images for training and validation
- **Model Training**:
  - Implements a CNN
  - Tracks training and validation loss/accuracy over epochs
- **Diagnostics**:
  - Generates ROC curves for each class
  - Saves predictions and validation labels to CSV files
  - Provides counts of actual/predicted galaxies for each class

#### Exercise 2 & Exercise 3: Regression Tasks 

The second and third exercise aim to predict the continuous values of the labels for questions Q2 and Q7 (exercise 2) and Q6 and Q8 (exercise 3) via a regression task.

- **Data Preprocessing**:
  - Extract columns for Q2 (`Class2.1`, `Class2.2`) and Q7 (`Class7.1`, `Class7.2`, `Class7.3`)
  - Use continuous label values
- **Model Training**:
  - Implements a CNN for regression
  - Tracks training loss over epochs
- **Diagnostics**:
  - Calculates Mean Squared Error (MSE) and Mean Absolute Error (MAE) for each class
  - Identifies the worst-performing class
  - Saves per-class errors and generates bar plots for MSE and MAE

---

## Project Structure Overview

The project is organized as follows:

```
galaxyzoo_Keusch/
│
├── data/                          # contains original and preprocessed data
│   ├── images/                    # original galaxy images
│   ├── training_solutions_rev1.csv # original labels for training
│   ├── Exercise1_ClassificationQ1  # data for 1_ClassificationQ1, created when running the script
│   ├── Exercise2_RegressionQ2Q7  # data for 2_RegressionQ2Q7, created when running the script
│   ├── Exercise3_RegressionQ6Q8  # data for 3_RegressionQ6Q8, created when running the script
│   ├── exercise_1/                # data for exercise 1, created when running the script
│   ├── exercise_2/                # data for exercise 2, created when running the script
│   └── exercise_3/                # data for exercise 3, created when running the script
│
├── galaxy_classification/         # core modules for the project
│   ├── data/                      # data loading and preprocessing
│   ├── networks/                  # neural network models
│   ├── utils/                     # utility functions
│   └── __init__.py
│
├── 1_ClassificationQ1                  # Script for Exercise 1 (Galaxy Classification)
├── 2_RegressionQ2Q7.py                  # Script for Exercise 2 (Regression for Q2 and Q7)
├── 3_RegressionQ6Q8.py                  # Script for Exercise 3 (Regression for Q6 and Q8)
│
├── Exercise_1.py                  # Script for Exercise 1 (Galaxy Classification)
├── Exercise_2.py                  # Script for Exercise 2 (Regression for Q2 and Q7)
├── Exercise_3.py                  # Script for Exercise 3 (Regression for Q6 and Q8)
├── train.py                       # Training script for training the classification task
├── evaluate.py                    # Evaluation script for classification
├── regression.py                  # Regression-specific utilities and models
└── README.md                      # Project documentation
```

The file "1_ClassificationQ1.py" can be run with the following comand:

```python
python 1_ClassificationQ1.py --run_name "runname" 
```

The files "exercise_1", "exercise_2", and "exercise_3" do not require a run name:

```python
   python Exercise_1.py
```

To train a model using the `train.py` script:
```bash
python train.py --run_name "run_name"
```

To evaluate a trained model using the `evaluate.py` script:
```bash
python evaluate.py --run_name "run_name"
```

## Prerequisites

To use this project, please ensure to have Python (Python 3.12 or higher) installed. Additionally, please run:

```
pip install -e .
```

---

## Disclaimer

#### Personal Struggles

Due to being unfamiliar with more complex project like the example in "project-template", and configuration files etc., I finitially tried to solve the exercises in simple script-files, which I uploaded as "Exercise_1.py", "Exercise_2.py" and "Exercise_3.py". Those can be run individually by using
```python
python Exercise_1.py 
```
and from those, the results presented in the written report are optained.

I wanted to use the provided project structure, however in all honestly I personally was struggling to change the given architecture and functions for the galaxyzoo project, and to implement my own functions, and to use the already existing training and evaluation functions and modules. 
I tried to do so in "1_ClassificationQ1.py" by using the "train.py" and "evaluate.py" functions, however the model performs really porly there and does not seem to learn or improve. 

Initially, I wanted to use the same "train.py" file for the training in the regression tasks, but faced challenges to adapt the training file to switch between the input and output requirements for the classification and regression tasks. Additionally, it would be more benefitial to move the evaluation for the regression tasks either into the "evaluation.py" file to take advantage of the architecture there, or create a separate evaluation file used for both regression tasks to avoid the unnecessary repetitions. 

#### Use of Generative Artificial Intelligence
 
Generative AI tools were used to assist in code debugging, documentation, and structuring the project. All code was reviewed and adapted to meet the specific requirements of the project.

