Here's a README for your evaluation models repository designed to guide users on how to use it effectively:

---

## Evaluation Models Repository for YOLOv8

### Overview
This repository is set up for evaluating YOLOv8 models on a specific dataset. It includes scripts for running predictions and computing evaluation scores.

### Prerequisites

### Getting Started



#### Step 1: Clone the Repository
Begin by cloning this repository to your local machine. Use the command:
```bash
git clone [URL to this repo]
cd [repo-name]
```

#### Step 2: Install dependencies

```bash
pip install -e requirements.txt
```

#### Step 3: Download and Set Up Data
1. **Download the Dataset**  
   Download the dataset `DS-71c1fd51-v2` from Google Drive:
   ```
   https://drive.google.com/file/d/17syKwltw8Jv-nYlLjzxbXQDd4p9hmkKC/view?usp=sharing
   ```
   Unzip the contents into the `Data` directory within this repository.

2. **Download Model Checkpoints**  
   Download the model checkpoints `cp` from Google Drive:
   ```
   https://drive.google.com/file/d/10zGvbR0nEvk5mPj7wEn3cHzWmyv4Hlqa/view?usp=sharing
   ```
   Unzip and place the folder in the same `Data` directory.

Your directory structure should now look like this:
```
repo-name/
│
└───Data/
    │
    ├───cp/
    │
    └───DS-71c1fd51-v2/
```

#### Step 4: Run Predictions
To compute predictions using the models and dataset, execute the following command:
```bash
python run_predictions.py
```

#### Step 5: Evaluate Results
After running predictions, evaluate the results using the Jupyter notebook provided:
```bash
jupyter notebook compute_scores.ipynb
```

#### Step 6: View erros using fiftyone
You can have a look to predictions erros using fiftyone running:
```bash
python fiftyone_create.py
```
