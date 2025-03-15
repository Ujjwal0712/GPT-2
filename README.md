Overview

The project implements a GPT-2 model, a transformer-based language model, from the ground up. The implementation focuses on optimizing training efficiency and memory usage while maintaining model performance. Key techniques used include:

- MatMul Precision: Utilizes mixed-precision matrix multiplication for faster computation and reduced memory usage.
- bfloat16: Employs the bfloat16 data type for training, which balances precision and range, enabling efficient training on hardware like TPUs and GPUs.
- Flash Attention: Integrates the Flash Attention mechanism to optimize the self-attention computation, reducing memory footprint and improving speed.
- Gradient Accumulation: Implements gradient accumulation to simulate larger batch sizes without increasing memory usage.

Features
- From-Scratch Implementation: The GPT-2 model is built without relying on high-level libraries like Hugging Face or PyTorch's built-in transformer modules.
- Efficient Training: Techniques like bfloat16, Flash Attention, and gradient accumulation ensure efficient training on limited hardware resources.
- Scalable: Designed to scale to larger datasets and model sizes with minimal modifications.
- Customizable: Easy to modify and extend for specific use cases or research experiments.

Dataset
The model is trained on the Cosmopedia-100k dataset, a high-quality, diverse dataset containing 100,000 text samples. The dataset is designed to provide a broad range of linguistic patterns and knowledge, making it ideal for training general-purpose language models.

The data processing pipeline downloads the Cosmopedia-100k dataset, tokenizes it using the GPT-2 tokenizer, and saves the tokenized data into shards for efficient training. Each shard contains 1 million tokens, and the pipeline uses multiprocessing to speed up tokenization.

Key Features of the Data Processing Script
- Tokenization: Uses the GPT-2 tokenizer from the tiktoken library.
- Sharding: Splits the dataset into shards of 100 million tokens each.
- Multiprocessing: Utilizes multiple CPU cores for parallel tokenization.
- Efficient Storage: Saves tokenized data as .npy files for fast loading during training.
