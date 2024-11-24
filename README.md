# Multi-Class Sentiment Analysis Using BERT

## Overview
This project showcases a comprehensive approach to **multi-class sentiment analysis** using BERT-based models, integrating technical and non-technical aspects of Natural Language Processing (NLP). The workflow includes data preprocessing, model training, evaluation, and prediction, offering an excellent demonstration of skills in NLP, machine learning, and data visualization.

---

## Key Features

### 1. **Data Preprocessing**
   - Handled **imbalanced data** with visualizations using Seaborn and Matplotlib.
   - Performed text tokenization using Hugging Face's `AutoTokenizer`.
   - Split dataset into training, validation, and testing sets, ensuring balanced label distributions.

### 2. **Model Training**
   - Utilized Hugging Face's `AutoModelForSequenceClassification` for fine-tuning BERT.
   - Customized model configuration to support multi-class classification with **dynamic label mapping**.
   - Employed `TrainingArguments` for defining training hyperparameters:
     - Batch size: 64
     - Learning rate: 2e-5
     - Epochs: 2
     - Evaluation and saving strategies: Per epoch.

### 3. **Evaluation**
   - Implemented metrics including **accuracy, F1-score, and recall** using scikit-learn and Hugging Face's `evaluate` library.
   - Visualized performance with a **confusion matrix heatmap**, showcasing predictions' effectiveness.

### 4. **Custom Prediction Pipeline**
   - Developed a function for real-time sentiment prediction with tokenization and inference on new texts.
   - Saved and exported the fine-tuned model for future use.

### 5. **Visualization**
   - Boxplots to analyze text length variations by sentiment category.
   - Bar plots to explore label distributions in the dataset.
   - Confusion matrix heatmap to assess model performance.

---

## Technical Skills Demonstrated
- **NLP Techniques**: Tokenization, sentiment analysis, text classification.
- **Modeling**: Fine-tuning BERT, dynamic configuration for label mapping.
- **Data Engineering**: Handling missing data, imbalanced classes, and stratified splitting.
- **Evaluation**: Metrics computation (F1, accuracy, recall) and visualization.
- **Python Libraries**: Transformers, Hugging Face, scikit-learn, Seaborn, Matplotlib.

## Non-Technical Skills Demonstrated
- **Data Storytelling**: Visualizations to interpret dataset and model outputs effectively.
- **Problem Solving**: Efficiently handling large datasets with reduced samples for faster experimentation.
- **Tool Optimization**: Leveraging cutting-edge tools like Hugging Face for faster model deployment.

---

## Code Highlights
### Real-Time Sentiment Prediction Function
```python
def predict_text_sentiment(text, model, tokenizer, device, labels):
    input_encoded = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_encoded)
    logits = outputs.logits
    predict = np.argmax(logits, axis=1)
    return predict, labels[predict[0]]
