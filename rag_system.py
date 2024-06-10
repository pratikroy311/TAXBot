# Import necessary libraries
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain

# Load and process documents
dir = "/content/data"

def load_docs(dir):
    loader = DirectoryLoader(dir)
    docs = loader.load()
    return docs

docs = load_docs(dir)

def split_docs(doc, chunk_size=512, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(doc)
    return docs

docs = split_docs(docs)

# Initialize embeddings and vector store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db"
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
vectordb.persist()
new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = new_db.similarity_search_with_score(query, k=k)
    else:
        similar_docs = new_db.similarity_search(query, k=k)
    return similar_docs

# Load LLM model from local directory
model_name = "./local_model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer
