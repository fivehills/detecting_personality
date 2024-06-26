# Big Five Personality Prediction

The models allow users to train and evaluate a model to predict Big Five personality traits from text data using Hugging Face Transformers.

## Features

- Preprocesses data for training
- Trains a multi-label regression model
- Evaluates model performance

## Cite:
```
article{wang2024personality,
  title={Continuous Output Personality Detection Models via Mixed Strategy Training},
  author={Rong Wang, Kun Sun},
  year={2024},
  journal={ArXiv},
  url={https://arxiv.org/abs/2406.16223}
}
```
## Installation

1. Clone the repository:
   ```
   git clone your-private-repository-url.git
   cd big5_personality_prediction
```

2. Install the required packages:
``
 pip install -r requirements.txt
```

3. Create a .env file in the root directory with your dataset URL:
```bash
DATA_URL=path/to/your/dataset.json
```

## File Structure:
```
big5_personality_prediction/
├── data/
│   └── sample_data.json  # Sample data placeholder
├── models/
│   └── __init__.py       # Initialization file for the models directory
├── utils/
│   ├── __init__.py       # Initialization file for the utils directory
│   ├── preprocess.py     # Data preprocessing
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── config.py             # Configuration settings
├── requirements.txt      # Required packages
├── .gitignore            # Ignored files and directories
└── README.md             # Project documentation
         
```
      
## Usage

###   Preprocess Data:
    
Ensure your dataset is in the correct format and update the DATA_URL in config.py.

 ```  
python utils/preprocess.py
```

###  Train Model:

Train the model using the preprocessed data.
```
python utils/train.py
```
### Evaluate Model:

Evaluate the trained model on the test dataset.
```
python utils/evaluate.py
```

## Configuration

Update config.py with the appropriate settings for your dataset and training parameters.

## License

This project is licensed under the MIT License.

php


## Initialization Files

To ensure that Python treats the `models` and `utils` directories as packages, add `__init__.py` files:

```
 models/__init__.py`

```

This file is intentionally left blank to indicate that this directory is a Python package.
```
utils/__init__.py

```
 This file is intentionally left blank to indicate that this directory is a Python package.

This structure and the simplified code should help you set up and run your project locally without exposing your code to the public.

### The model is available at: https://huggingface.co/KevSun/Personality_LM
