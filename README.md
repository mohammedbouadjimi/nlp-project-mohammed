# Bot or Not? Detecting Machine-Generated Tweets

### Author: Mohammed Bouadjimi  
### Course: NLP Assignment 2 – August 2025

---

##  Problem Definition

The increasing sophistication of AI language models like ChatGPT raises growing concerns over the authenticity of online content, especially on platforms such as Twitter, where both human and machine-generated content co-exist. This project addresses the **binary classification task** of detecting whether a tweet is written by a human or by a machine.

The ability to distinguish human-written content from AI-generated text is crucial for:
- Detecting misinformation and bot-generated propaganda
- Preserving the integrity of public discourse
- Enabling platforms to flag or filter synthetic content

We frame this as a **supervised classification problem**, where the model learns to predict:
- `0`: Human-written tweet  
- `1`: Machine-generated tweet (e.g., by ChatGPT)

---

##  Dataset Used

The dataset was sourced from the following shared Google Drive folder:  
[Dataset Folder](https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc)

### Dataset Details:
- **Human-written tweets**: Scraped from real Twitter users.
- **Machine-generated tweets**: Generated using OpenAI's ChatGPT, mimicking real tweet formats.
- Each row includes:
  - `clean_text`: preprocessed version of the tweet
  - `label`: 0 for human, 1 for bot

To optimize training time and reduce computational cost:
- We sampled **10,000 training examples**
- And **2,000 validation examples**

All tweets were in English and underwent cleaning to remove emojis, special characters, and URLs.

---

##  Evaluation Metrics

To assess model performance, we used:
- **Accuracy**: Measures overall correctness of the model
- **F1 Score**: Harmonic mean of precision and recall, especially useful for imbalanced datasets
- **Validation Loss**: Cross-entropy loss on the dev set, used to monitor overfitting

These metrics provide a balanced understanding of the model’s strengths and weaknesses.

---

##  Model Architecture and Training Pipeline

We used **RoBERTa-base**, a transformer-based model pretrained on a large corpus of English text.

### Pipeline Overview:
1. **Tokenization**:
   - Used Hugging Face's `AutoTokenizer` with padding/truncation (max length: 128 tokens)
2. **Dataset Preparation**:
   - Converted `pandas` DataFrames into Hugging Face `Dataset` objects
3. **Model Setup**:
   - Used `RobertaForSequenceClassification` with 2 output classes
   - Fine-tuned using `Trainer` API from `transformers`
4. **Training Configuration**:
   - Optimizer: AdamW
   - Batch size: 16
   - Epochs: 3
   - Evaluation strategy: After each epoch
   - GPU: Enabled via Google Colab

This configuration provided a balance between speed and model performance.

---

##  Results Achieved

Below are the evaluation metrics after 3 training epochs:

| Metric            | Value        |
|-------------------|--------------|
| **Accuracy**      | 67.2%        |
| **F1 Score**      | 0.666        |
| **Val. Loss**     | 1.867        |
| **Training Time** | ~15 seconds/epoch |

These results show that the model has learned meaningful patterns to distinguish between human and AI text, outperforming a random guess baseline (~50% accuracy). However, there is room for further optimization.

---

##  Comparison with Baseline

No baseline models were explicitly implemented in this version, but the results can be evaluated against:
- **Random Classifier**: Expected accuracy = 50%
- **Majority Class Classifier**: Also performs poorly due to near balance in class distribution

The RoBERTa model achieved **~67% accuracy**, indicating that it captures non-trivial linguistic or stylistic differences between humans and bots.

In future iterations, we plan to compare against:
- Logistic Regression with TF-IDF
- LSTM/BiLSTM models
- ChatGPT-based prompt-only inference

---


---

## 🔍 Future Work

This project lays the foundation for more advanced bot detection systems. Potential improvements include:
- Dataset expansion with more varied tweet topics
- Hyperparameter tuning (e.g., learning rate schedulers)
- Adding explainability using SHAP or attention visualizations
- Exploring zero-shot or few-shot classification using larger models like `roberta-large` or `chatglm`


---

##  Dataset Reference

Google Drive shared folder:  
https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc

---

##  Acknowledgments

- Hugging Face  for the `transformers` and `datasets` libraries
- Google Colab for providing free GPU access
- Algonquin College – NLP course (Fall 2025)

---



