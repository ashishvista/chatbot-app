from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

class ChatbotModel:
    def __init__(self, model_name: str, vector_store_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.vector_store = FAISS.load_local(vector_store_path, HuggingFaceEmbeddings(model_name))

    def generate_response(self, query: str) -> str:
        inputs = self.tokenizer.encode(query, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def retrieve_and_generate(self, query: str) -> str:
        retriever = self.vector_store.as_retriever()
        docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in docs])
        full_query = f"{context}\n\n{query}"
        return self.generate_response(full_query)

# Example usage:
# model = ChatbotModel(model_name="gpt-3.5-turbo", vector_store_path="path/to/vector/store")
# response = model.retrieve_and_generate("What is the capital of France?")
# print(response)