import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import pandas as pd

# Load and display logo
logo_url = "logo.png"  # Replace with the URL of your logo or local file path
st.image(logo_url, width=200)

# Sidebar input for OpenAI API Key
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Please Enter your OpenAI API key",
    type="password"
)

if not user_api_key:
    st.sidebar.error(
        "API key is required. Get your OpenAI API key [here](https://platform.openai.com/account/api-keys)"
    )

# Sidebar file uploader for CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# Main logic is executed when a file is uploaded
if uploaded_file:
    # Temporary file storage
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()

    # Display data under the logo
    st.write("### Dataset Preview")
    st.dataframe(pd.read_csv(tmp_file_path))

    # Initialize embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)  
    vectors = FAISS.from_documents(data, embeddings)

    # Initialize conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0.0,
            model_name='gpt-3.5-turbo',
            openai_api_key=user_api_key
        ),
        retriever=vectors.as_retriever()
    )

    # Function for handling conversational chat
    def conversational_chat(query):
        result = chain({
            "question": query,
            "chat_history": st.session_state.get('history', [])
        })
        st.session_state.setdefault('history', []).append((query, result["answer"]))
        return result["answer"]

    # Initialize session state if not present
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["I am your analytical assistant, here to provide insights and analysis at your request. "])
    st.session_state.setdefault('past', ["Greetings!"])

    # Containers for chat history and user input
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            # User input field and submit button
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (: ", key='input')
            submit_button = st.form_submit_button(label='Send')

        # Process user input on submit
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                st.write(f"User: {st.session_state['past'][i]}")
                st.write(f"Chatbot: {st.session_state['generated'][i]}")

    # Compile chat history into a string
    chat_history_str = "\n".join(
        f"User: {st.session_state['past'][i]}\nBot: {st.session_state['generated'][i]}"
        for i in range(len(st.session_state['generated']))
    )

    # Create a download button for exporting chat history
    st.download_button(
        label="Download Chat History",
        data=chat_history_str,
        file_name="chat_history.txt",
        mime="text/plain"
    )
st.sidebar.markdown("""
**Disclaimer:**

This product is developed and distributed by AnalytiXplore. It is provided "as is" without warranty of any kind, either express or implied, including but not limited to the implied warranties of merchantability or fitness for a particular purpose.

AnalytiXplore does not collect, store, process, or manage any data accessed or generated through the use of this product. All data handling, storage, and management are the sole responsibility of the user or the user's organization. AnalytiXplore shall have no responsibility or liability with respect to any data privacy laws, regulations, or standards that may be applicable.

Users are strongly advised to review their data privacy practices and ensure compliance with all relevant laws and regulations. It is the user’s responsibility to secure any necessary consents, permissions, or authorizations required for the collection, processing, and storage of data.

Use of this product signifies acceptance of these terms and an understanding that AnalytiXplore bears no responsibility for any data privacy concerns arising from the use of this product.
""", unsafe_allow_html=True)

if st.sidebar.button('Contact Us'):
    st.sidebar.write("[Send Email](mailto:info@analytixplore.com)")

# Run with: streamlit run tuto_chatbot_csv.py
