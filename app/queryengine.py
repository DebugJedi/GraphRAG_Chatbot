from langchain_core.prompts import PromptTemplate
from typing import List, Tuple, Dict
import heapq
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Dict


class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer to the query")
    answer: str = Field(description="The current answer based on the context, if any")

class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_content_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()
    
    def _create_answer_check_chain(self):
        """
        Creates a chain to check if the context provides a complete answer to the query.

        Args:
        - None

        Returns:
        - Chain: A chain to check if the context provides a complete answer.
        """

        answer_check_prompt = PromptTemplate(
            input_variables=['query', 'context'],
            template="Given the query: '{query}' And the current context: {context} Does this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete. Is complete answer (Yes/No): Answer (if complete):"
            )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)
    
    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Checks if the current context provides a complete answer to the query.

        Args:
        - query (str): The query to be answered.
        - context (str): The current context.

        Returns:
        - tuple: A tuple containing:
            - is_compelet (bool): Whether the context provides a complete answer.
            - answer (str): The answer based on the context, if complete.
        """
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer
    
    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        Expands the context by traversing the knowledge graph using a Dijkstra-like approach.

        This method implements a modified version of Dijkstra's algorithm to explore the knowledge graph.
        prioritizing the most relevant and strogly connected information. The algorithm works as follows:
        
        1. Initialize:
            - Start with nodes corresponding to the most relevant documents.
            - Use a priority queue to manage the traversal order, where priority is based on connection strength.
            - Maintain a dictionary of best known "distances" (invesrse of connection strengths) to each node.

        2. Traverse:
            - Always exploere the node with the highest priority (strogest connection) next.
            - For each node, check if we've found a complete answer.
            - Explore the nodes's neighbors, updating their priorities if a stronger connection is found.

        3. Concept Handling:
            - Track visited concepts to guide the exploration towards new, relevant information.
            - Expand to neighbors only if they introduce new concepts.

        4. Termination:
            - Stop if a complete answer is found.
            - Continure until the priority queue is empty (all reachable nodes explored).
        This approach ensures that:
        - We prioritize the most relevant and strogly connected information.
        - We explore new concepts systematically.
        - We find the most relevant answer by following the strongest connections in the knowledge graph.

        Args:
        - query (str): The query to be answered.
        - relevant_docs (List[Document]): A list of relevant documents to start the traversal.

        Returns:
        - tuple: A tuple containing:
            - expanded_context (str): The accumulated context from traversed nodes.
            - tranversal_path (List[int]): The sequence of node indices visited.
            - filtered_content (Dict[int, str]): A mapping of node indices to their content.
            - final_answer (str): The final answer found, if any.
        """
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}
        print("\nTraversing the knowledge graph:")

        for doc in relevant_docs:
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            priority = 1/similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node]= priority
        step=0
        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)

            if current_priority> distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step +=1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # st.write(f"<span style='color:red;'>Step {step} - Node {current_node}:</span>", unsafe_allow_html=True)
                # st.write(f"Content: {node_content[:100]}....")
                # st.write(f"Concepts: {', '.join(node_concepts)}")
                print("-"*50)

                # Check if we have a complete answer with the current context
                is_complete, answer  = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break
                node_concepts_set = set(self.knowledge_graph._lemmatize_concepts(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # Calcualte new distance (priority) to the neighbor
                        distance = current_priority + (1/edge_weight)

                        # If we've found a stronger connection to the neighor, update its distance
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            if neighbor not in traversal_path:
                                step+=1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context +="\n"+neighbor_content if expanded_context else neighbor_content
                                
                                # Log the neighbor node information
                                # st.write(f"<span style='color:red;'>Step {step} - Node {neighbor} (neighbor of {current_node}):</span>", unsafe_allow_html=True)
                                # st.write(f"Content: {neighbor_content[:100]}...")
                                print(f"Concepts: {', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break
                                neighbor_concepts_set = set(self.knowledge_graph._lemmatize_concepts(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                if final_answer:
                    break
        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query. Context: {context} \n Query: {query} Answer:"
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)
        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        Processes a query by retrieving relevant docments, expanding the context, and generating the final answer.

        Args:
        - query (str): The query to be answered.

        Returns:
        - tuple: A tuple containing:
            - final_answer (str): The final answer to the query.
            - traversal_path (list): The traversal path of nodes in the knowledge graph.
            - filtered_content (dict): The filtered content of nodes.
        """
        with get_openai_callback() as cb:
            st.write(f"\nProcessing query: {query}")
            relevant_docs = self._retrieve_relevant_documents(query)
            expanded_context, tranversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)

            if not final_answer:
                # st.write("\nGenerating final answer....")
                response_prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template="Based on the following context, please answer the query. Context: {context} Query: {query} Answer:"
                )
                response_chain = response_prompt | self.llm
                input_data = {"query":query, "context": expanded_context}
                response = response_chain.invoke(input_data)
                final_answer = response.content
            else:
                pass
                # st.write("\nComplete answer found during traversal.")
            # st.write(f"\nFinal Answer: {final_answer}")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): {cb.total_cost}")
        return final_answer, tranversal_path, filtered_content

    def _retrieve_relevant_documents(self, query:str):
        """
        Retrieves relevant documents based on the query using the vector store.

        Args:
        - list: A list of relevant documents.
        """            
        print("\nRetrieving relevant documents....")
        retiever = self.vector_store.as_retriever(search_type="similarity", search_kwargs= {"k":5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retiever)
        return compression_retriever.invoke(query)
    

    

                        