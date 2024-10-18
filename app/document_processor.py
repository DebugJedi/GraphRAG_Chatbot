from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class DocumentProcessor:
    def __init__(self):
        """
        Initializes the DocumentProcessor with a text splitter and OpenAI embeddings.

        Attributes:
        - text_splitter: An instance of RecursiveCharacterTextSplitter with specified chunk size and overlap.
        - embeddings: An instance of OpenAIEmbeddings used for embedding documents.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,
                                                        chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings(api_key=st.secrets["API_KEY"])

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into smaller chunks and a vector store.
        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - tuples: A tuple containing:
            - splits (list of str): The list of split document chunks.
            - vector_store (FAISS): A FAISS vector store created from the split document chunks and their embeddings.
        """
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store
    
    def create_embeddings_batch(self, texts, batch_size=32):
        """
        Creates embeddings for a list of texts in batches.

        Args:
        - texts (list of str): A list of texts to be embedded.
        - batch_size (int, optional): The number of texts to process in each batch.
        
        Returns:
        - numpy.ndarrays: An array of embeddings for hte input texts.
        """

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def compute_similarity_matrix(self, embeddings):
        """
        Computes a cosine similarity matrix for a given set of embeddings.
        Args:
        - embeddings (numpy.ndarray): An array of embeddigs.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the embeddding provided.
        """
        return cosine_similarity(embeddings)