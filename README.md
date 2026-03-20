# 🤖 Custom GPT-2 RAG: Fine-Tuned WikiText-2 Assistant

This repository contains a complete **Retrieval-Augmented Generation (RAG)** pipeline. It features a custom-configured **GPT-2** model fine-tuned on the WikiText-2 dataset, integrated with a **FAISS** vector database for context-aware question answering.



---

## 🚀 Key Features
* **Custom Fine-Tuning:** GPT-2 model trained on `wikitext-2-raw-v1` using the Hugging Face `Trainer` API.
* **Vector Search:** Integrated **FAISS** (Facebook AI Similarity Search) for high-speed document retrieval.
* **Optimized Context Window:** Implemented **recursive chunking** and **prompt truncation** to handle GPT-2's 1024-token limitation.
* **Semantic Embeddings:** Uses `all-MiniLM-L6-v2` from Sentence-Transformers to map queries to document context.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language Model** | GPT-2 (Fine-tuned) |
| **Retriever** | FAISS (L2 Distance) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Frameworks** | PyTorch, Hugging Face Transformers, Datasets |
| **Data** | WikiText-2 Raw |

---

## 📖 How It Works

### 1. Training Phase
The model is a custom GPT-2 configuration ($L=6, H=6, E=384$) designed to be lightweight. It was fine-tuned for **2 epochs** to adopt a formal, encyclopedic linguistic style suitable for technical queries.

### 2. Retrieval Phase
When a user submits a query:
1.  **Embedding:** The query is converted into a 384-dimensional vector.
2.  **Search:** FAISS identifies the `Top-K` most relevant text chunks from the indexed documents.
3.  **Augmentation:** These chunks are injected into a structured prompt: `Context: ... Question: ... Answer:`.

### 3. Generation Phase
The fine-tuned LLM processes the augmented prompt. Using `top_p` and `temperature` sampling, it generates a coherent answer based strictly on the retrieved data.



---

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/rag-project.git](https://github.com/yourusername/rag-project.git)
    cd rag-project
    ```
2.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch transformers datasets faiss-cpu sentence-transformers accelerate
    ```
4.  **Run the RAG system:**
    ```bash
    python rag.py
    ```

---

## 🧠 Challenges Overcome
* **Context Window Limits:** Resolved `IndexError: index out of range` by implementing a hard-limit truncation at the tokenizer level (1024 tokens).
* **Windows Multiprocessing:** Wrapped the training loop in `if __name__ == "__main__":` to ensure compatibility with Windows process spawning.
* **Resource Optimization:** Reduced GPT-2 layers from 12 to 6 to allow for successful training on consumer-grade CPU hardware.

---

## 🎯 Future Roadmap
* Implement **Sliding Window Chunking** with overlap to preserve context at boundaries.
* Deploy a **Gradio Web UI** for a more interactive user experience.
* Experiment with **Quantization (8-bit)** to further reduce memory usage.
