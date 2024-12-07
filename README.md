# Xitsonga Text Generation using LSTM

## Problem Statement
This project develops a **text generation model** for the **Xitsonga** language using the **LSTM** architecture, trained on a corpus from the **Sadiler website** to generate coherent Xitsonga text.

## Key Steps

1. **Data Collection**: Corpus collected from the **Sadiler website**.
2. **Data Cleaning**: Remove numbers, special characters, and whitespace, and convert text to lowercase.
3. **EDA**: Analyze word frequency and sentence length.
4. **Preprocessing**: Tokenize text, create sequences, and pad them.
5. **Model Building**: Build LSTM model with embedding, LSTM, dropout, and dense layers.
6. **Training**: Split data into training, validation, and test sets. Train with **Adam optimizer** and **categorical cross-entropy** loss.
7. **Text Generation**: Generate text with temperature-controlled sampling based on seed text.
8. **Evaluation**: Use loss and perplexity for model evaluation.

## Conclusion
A text generation model for **Xitsonga** was built using **LSTM**. It can be used for conversational agents or content generation in Xitsonga.

## Technologies Used
- **TensorFlow,Keras**, **Matplotlib**, **NumPy**, **Pandas**


