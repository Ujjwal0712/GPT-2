GPT-2 From Scratch: Efficient and Scalable Implementation 🚀

Welcome to the GPT-2 From Scratch project! This repository contains a ground-up implementation of the GPT-2 model, a transformer-based language model, with a focus on optimizing training efficiency and memory usage while maintaining high performance.

🚀 Key Features

✅ Model Implementation

- From-Scratch Implementation: Built without relying on high-level libraries like Hugging Face or PyTorch's built-in transformer modules.

- Efficient Training: Leverages advanced techniques to ensure efficient training on limited hardware resources:

- Mixed-Precision MatMul: Uses mixed-precision matrix multiplication for faster computation and reduced memory usage.
 
- bfloat16: Balances precision and range for efficient training on TPUs and GPUs.

- Flash Attention: Optimizes self-attention computation, reducing memory footprint and improving speed.

- Gradient Accumulation: Simulates larger batch sizes without increasing memory usage.

- Scalable: Designed to scale to larger datasets and model sizes with minimal modifications.

- Customizable: Easy to modify and extend for specific use cases or research experiments.

📂 Dataset: Cosmopedia-100k

The model is trained on Cosmopedia-100k, a high-quality, diverse dataset containing 100,000 text samples. This dataset provides a broad range of linguistic patterns and knowledge, making it ideal for training general-purpose language models.

🔧 Data Processing Pipeline

The data processing pipeline is optimized for efficiency and speed. Here's how it works:

- Tokenization: Uses the GPT-2 tokenizer from the tiktoken library.

- Sharding: Splits the dataset into shards of 1 million tokens each for efficient handling.

- Multiprocessing: Utilizes multiple CPU cores for parallel tokenization, significantly speeding up the process.

- Efficient Storage: Saves tokenized data as .npy files for fast loading during training.
