

# EthioMart: Amharic Named Entity Recognition (NER) System for Telegram E-Commerce

EthioMart is an initiative to consolidate multiple Ethiopian Telegram-based e-commerce channels into one centralized platform. This project focuses on developing a fine-tuned **Amharic NER system** that extracts key business entities such as product names, prices, and locations from Telegram messages. The extracted data will help populate EthioMart's centralized database.

## Project Overview

This repository contains the necessary code and documentation for developing an Amharic NER system, which includes tasks like data ingestion, preprocessing, model fine-tuning, and comparison.

### Key Objectives

1. Real-time data extraction from multiple Ethiopian Telegram channels.
2. Fine-tuning pre-trained models for extracting entities such as:
   - Product names (e.g., "ተሸክላ" - Cup)
   - Prices (e.g., "100 ብር" - 100 birr)
   - Locations (e.g., "አዲስ አበባ" - Addis Ababa)
3. Model evaluation and comparison to select the best-performing NER model.

## Project Structure

```bash
├── data/                      # Preprocessed and raw dataset files
├── models/                    # Saved models after fine-tuning
├── notebooks/                 # Jupyter notebooks for data analysis and model training
├── scripts/                   # Scripts for data preprocessing, training, and evaluation
└── README.md                  # Project documentation
```

### Files of Interest

- **`data_ingestion.py`**: Fetches messages and data from relevant Telegram channels.
- **`preprocessing.py`**: Preprocesses Amharic text data (tokenization, normalization).
- **`fine_tune_ner.py`**: Fine-tunes the NER models using Hugging Face's Trainer API.
- **`evaluate_model.py`**: Evaluates and compares different NER models for performance.
- **`ner_data_conll.txt`**: Labeled dataset in CoNLL format for model training.

## Getting Started

### Prerequisites

To run this project, you will need:

- Python 3.8 or higher
- Libraries: 
  - Hugging Face Transformers
  - PyTorch
  - Pandas
  - Datasets (Hugging Face)
  - Tokenizers
  - SHAP, LIME (for model interpretability)

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Setup and Usage

1. **Data Ingestion**
   - Use `data_ingestion.py` to collect data from multiple Telegram channels. You will need to set up a Telegram API and provide credentials for access.
   
   ```bash
   python scripts/data_ingestion.py
   ```

2. **Data Preprocessing**
   - Run `preprocessing.py` to tokenize and normalize the Amharic text and structure the data for NER tasks.
   
   ```bash
   python scripts/preprocessing.py
   ```

3. **Labeling for NER**
   - Label the dataset using the CoNLL format. An example of the format is available in `ner_data_conll.txt`.

4. **Model Fine-Tuning**
   - Fine-tune models like XLM-Roberta, DistilBERT, or mBERT by running the `fine_tune_ner.py` script. Adjust hyperparameters like epochs, learning rate, and batch size as needed.
   
   ```bash
   python scripts/fine_tune_ner.py
   ```

5. **Model Evaluation**
   - Compare multiple models for accuracy, speed, and robustness using `evaluate_model.py`. It will generate a comparison report based on key evaluation metrics.
   
   ```bash
   python scripts/evaluate_model.py
   ```

6. **Model Interpretability**
   - Use SHAP and LIME tools to interpret how the models identify entities and generate reports for transparency.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
