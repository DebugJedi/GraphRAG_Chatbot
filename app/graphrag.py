from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from app.document_processor import DocumentProcessor
from app.knowledgegraph import knowledgeGraph
from app.visualizer import Visulaizer
from app.queryengine import QueryEngine
import streamlit as st

class GraphRAG:
    def __init__(self):
        """
            Initializes the GraphRAG system with components for document processing, knowledge graph construction,
            querying and visualization.

            Attributes: 
            - llm: An instance of large language model (LLM) for genrating responses.
            - embedding_model: An instance of embedding model for document embeddings.
            - documents_processor: An instance of DocumentProcessor class for processing documents.
            - knowledge_graph: An instance of the knowledgeGraph class for building and manging knowledge graph.
            - query_engine: An instance of the QueryEngine class for handling queries.
            - visulaizer : An instance of the Visuliazer class for visualizing the knowledge graph.
        """
        self.llm = ChatOpenAI(temperature=0, model_name = "gpt-4o-mini", max_tokens= 3000, api_key=st.secrets["API_KEY"])
        self.embedding_model = OpenAIEmbeddings(api_key=st.secrets["API_KEY"])
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = knowledgeGraph()
        self.query_engine = None
        self.visualizer = Visulaizer()

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into chunks, embedding them, and building a knowledge graph.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - None
        """
        splits, vector_store = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm)

    def query(self, query: str):
        """
        Handles a query by retrieving relevant information from the knowledge graph and visulaizing the traversal path.

        Args:
        - query (str): The query to be answered.

        Returns:
        - str: The response to the query.
        """
        response, traversal_path, filtered_content = self.query_engine.query(query)

        # if traversal_path:
        #     self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        # else:
        #     print("No traversal path to visulaize.")

        return response

