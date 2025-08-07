# RAG-based-object-Retreival for Pizza Restaurant Reviews


This project implements a **Retrieval-Augmented Generation (RAG)** system to answer questions about a pizza restaurant based on customer reviews. It uses **LangChain**, **Ollama**, and **Chroma** to retrieve relevant data and generate responses using a language model.

---

## 🔧 Features

- RAG pipeline with **LLaMA3.2** via Ollama
- Vector search using **Chroma** and `mxbai-embed-large` embeddings
- Realistic restaurant reviews as a knowledge base
- Simple interactive CLI chatbot

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/Santhosh-r7/RAG-based-object-Retreival
cd RAG-based-object-Retreival
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Pull required Ollama models:

```bash
ollama run llama3
ollama run mxbai-embed-large
```

---

## 🚀 Usage

```bash
python main.py
```

Example interaction:

```
Enter question for the chatbot:(type q to exit)
> What are customers saying about the crust?
```

The system retrieves top 2 relevant reviews using Chroma and uses them to answer your query using LLaMA3.

---

## 🧾 Dataset

The `realistic_restaurant_reviews.csv` should have the following columns:

- `Title`
- `Review`
- `Rating`
- `Date`

These are used to generate vector embeddings for retrieval.

---

## 📁 File Structure

```
├── main.py                    # RAG interaction loop
├── vector.py                  # Chroma DB creation + retriever logic
├── realistic_restaurant_reviews.csv  # Review dataset
├── requirements.txt           # Python dependencies
└── chroma_db/                 # Chroma DB (generated)
```

---

## 🛠️ Tech Stack

- Python
- LangChain
- Ollama
- ChromaDB
- Pandas

---

## 📝 License

MIT License
