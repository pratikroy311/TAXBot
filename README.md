# TaxBot: A RAG Application for Indian Taxation

TaxBot is a Retrieval-Augmented Generation (RAG) application designed to assist users with queries related to Indian taxation. It combines state-of-the-art machine learning models and retrieval techniques to deliver accurate and context-aware responses.

## Features
- **RAG Architecture**: Combines retrieval and generation to enhance response accuracy.
- **Domain-Specific Expertise**: Focused on Indian taxation laws and policies.
- **Technologies Used**: Built using LangChain, ChromaDB, and Zephyr-7b-beta model from Hugging Face.
- **Efficient Knowledge Retrieval**: Integrates a custom-built knowledge base for precise information extraction.

## Tech Stack
- **LangChain**: Framework for building applications powered by language models.
- **ChromaDB**: Vector database for managing and retrieving embeddings.
- **Hugging Face Zephyr-7b-beta**: A fine-tuned large language model optimized for Indian taxation.

## Installation

### Prerequisites
- Python 3.8+
- Pip package manager
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/taxbot.git
   cd taxbot
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # For Linux/macOS
   venv\Scripts\activate   # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables:
   - Create a `.env` file in the root directory.
   - Add the required API keys and configuration settings. Example:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     CHROMADB_PATH=./data/chromadb
     ```
5. Run the application:
   ```bash
   python app.py
   ```

## Usage
- Open your browser and navigate to the local server URL (e.g., `http://127.0.0.1:5000`).
- Input your taxation-related query, and TaxBot will provide detailed responses.

## Directory Structure
```
TaxBot/
├── app.py               # Main application entry point
├── requirements.txt     # Python dependencies
├── data/                # ChromaDB data storage
├── models/              # Model-related files
├── utils/               # Helper functions and utilities
└── README.md            # Project documentation
```

## Future Enhancements
- Expand the knowledge base to include more domains beyond taxation.
- Enhance multi-language support for Indian regional languages.
- Integrate with external APIs for live updates on taxation rules.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- [Hugging Face](https://huggingface.co) for the Zephyr-7b-beta model.
- [LangChain](https://langchain.com) for the framework support.
- [ChromaDB](https://www.trychroma.com) for efficient vector database management.


#### NOTE: You can run this notebook as well as chatbot in google colab. As long as colab is running the model will be hosted and link can be shared with others
