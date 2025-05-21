# DL_Assignment3

---
## Seq2Seq Model with PyTorch :

This repository provides an implementation of a Sequence-to-Sequence (Seq2Seq) model using PyTorch. Seq2Seq models are commonly applied to tasks like machine translation, text summarization, and speech recognition, where the goal is to transform one sequence into another.

---
### Introduction :
The Seq2Seq model is composed of two main components: an encoder and a decoder. The encoder reads the input sequence and transforms it into a fixed-size context vector that captures the semantic meaning of the entire input. The decoder then uses this context vector to generate the corresponding output sequence step by step.

---

### Overview

The goal of this assignment is to explore sequence-to-sequence learning using Recurrent Neural Networks (RNNs). The work is divided into four main parts:

1. **Modeling Seq2Seq Tasks**: Understand and implement the core idea behind sequence-to-sequence models using RNNs.
2. **Cell Comparison**: Experiment with different RNN variants—vanilla RNN, LSTM, and GRU—and compare their performance.
3. **Attention Mechanism**: Dive into attention-based models to see how they address the limitations of basic seq2seq architectures, especially for longer sequences.
4. **Visualization**: Visualize how different components in the RNN-based model interact during training and inference, providing deeper insight into model behavior.
---
**Encoder**
The `Encoder` class is designed to encode an input sequence into a context representation that can be used by the decoder. It is initialized with the following parameters:

* **input\_size**: Size of the input vocabulary.
* **embedding\_size**: Dimensionality of the word embeddings.
* **hidden\_size**: Number of units in the hidden state of the recurrent layer.
* **num\_layers**: Number of stacked recurrent layers.
* **dropout**: Dropout rate used to prevent overfitting.
* **cell\_type**: Type of recurrent cell to use — options include RNN, LSTM, or GRU.
* **bidirectional**: Boolean flag indicating whether the encoder should process the sequence in both forward and backward directions.

The `forward` method of the encoder processes the input sequence and outputs the final hidden state(s), which summarize the information in the entire input and are passed to the decoder for generating the output sequence.
---
**Decoder**
The `Decoder` class is responsible for generating the output sequence using the context information provided by the encoder. It is initialized with the following parameters:

* **output\_size**: Size of the output vocabulary.
* **embedding\_size**: Dimensionality of the word embeddings.
* **hidden\_size**: Number of units in the hidden state of the recurrent layer.
* **num\_layers**: Number of stacked recurrent layers.
* **dropout**: Dropout rate to help prevent overfitting.
* **cell\_type**: Type of recurrent cell to use — RNN, LSTM, or GRU.

The `forward` method of the decoder takes the encoder’s final hidden state(s) as the initial context and generates the output sequence one token at a time. It uses the hidden states and previously generated outputs to predict the next token in the sequence.
---
**Seq2Seq Model**
---
The `Seq2Seq` class integrates both the encoder and decoder to build the complete sequence-to-sequence architecture. It is initialized with an encoder and a decoder, enabling it to perform end-to-end sequence translation.

The `forward` method accepts a source sequence and a target sequence as input. It first encodes the source sequence using the encoder and then uses the decoder to generate the predicted output sequence based on the encoded context and the target inputs.
---
**Performance**

* Using the best model from the sweep without attention—GRU cell, 3-layer encoder, 2-layer decoder, 256 hidden size, 16 embedding size, 0.3 dropout, and 6 epochs—we achieved strong performance. The model reached 32.3% training accuracy and 34.0% validation accuracy, with corresponding losses of 0.36.

 
* Using attention with the best hyperparameters—LSTM cell, bidirectional encoder, 3 encoder layers, 2 decoder layers, 128 hidden and embedding size, 0.3 dropout, and 10 epochs—the model showed significant improvement. It achieved 47.68% training accuracy, 38.09% validation accuracy, and 37.9% test accuracy (exact word match). This demonstrates the effectiveness of attention in enhancing transliteration performance.

