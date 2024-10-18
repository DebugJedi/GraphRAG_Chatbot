import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from app.graphrag import GraphRAG
from streamlit_chat import message
import json
from langchain.schema import AIMessage

# "Chat bot using local RagGraph 🕸️🦜"

def is_json(obj):
    try:
        json.loads(obj)
        return True
    except ValueError:
        return False

def main():
    st.title("PDF Query App: Instant Answers at Your Fingertips 📄🔍⚡")
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False
    
    uploaded_file = st.file_uploader("Upload you PDF here 👇: ", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing...."):
            with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            documents = documents[:10]
            st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcom! you can now ask any question regarding "+ uploaded_file.name]
        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        response_container = st.container()

        container = st.container()

        with container:
            with st.form(key = 'my_form', clear_on_submit=True):
                query = st.text_input("Enter your question: ", key= 'input')
                submit_button = st.form_submit_button(label='Send')
            if submit_button and query:
                with st.spinner("Processing...."):
                    graph_rag = GraphRAG()
                    graph_rag.process_documents(documents)
                    output = graph_rag.query(query)
                    if output is not None:
                        # st.write("AIMessage Output:", output)
                        # st.write("AIMessage Output:", output.content)
                        if hasattr(output, 'content'):
                            # st.write(output.content)
                            response_text = output.content
                        elif hasattr(output, 'text'):
                            # st.write("checking for text.......")
                            st.write("#"*50)
                            response_text = output.text
                        else:
                            # st.write("str object.......")
                            response_text =output
                        # response_text = output.content

                        st.session_state.past.append(query)
                        st.session_state.generated.append(response_text)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
    
if __name__ == '__main__':
    main()
    