import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# --- Configuration ---
# (Preferably store these as secrets in Streamlit Cloud or a .env file)
ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)

# --- Helper Functions ---
def load_pdf(uploaded_file):
    raw_text = ''
    pdfreader = PdfReader(uploaded_file)
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def add_to_vector_store(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    vector_store.add_texts(texts)  # Add all texts

# --- Streamlit App ---
st.title("DataScience:GPT - PDF Q&A Chatbot")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    raw_text = load_pdf(uploaded_file)
    add_to_vector_store(raw_text)
    st.success("PDF processed and added to the vector store!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Ask me questions about your uploaded PDF!"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
        answer = vector_index.query(prompt, llm=llm)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Accuracy Testing ---
if st.button("Run Accuracy Test"):
    test_data = pd.read_csv("test.csv")
    
    # Display the test data to verify the column names
    st.write(test_data.head())
    
    # Verify column names
    required_columns = ["question", "answer"]
    for column in required_columns:
        if column not in test_data.columns:
            st.error(f"Missing required column: {column}")
            st.stop()

    predictions = []
    true_answers = test_data["answer"].tolist()

    vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

    for question in test_data["question"]:
        prediction = vector_index.query(question, llm=llm)
        predictions.append(prediction)

    # Calculate semantic similarity
    true_embeddings = model.encode(true_answers, convert_to_tensor=True)
    pred_embeddings = model.encode(predictions, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(true_embeddings, pred_embeddings)

    threshold = 0.7  # Define a threshold for similarity
    correct_predictions = (similarities.diag() > threshold).sum().item()
    accuracy = correct_predictions / len(true_answers)

    st.write(f"Accuracy: {accuracy}")

    # Calculate F1 score for exact match comparison
    true_answers = [preprocess_text(answer) for answer in true_answers]
    predictions = [preprocess_text(prediction) for prediction in predictions]

    f1 = f1_score(true_answers, predictions, average="weighted")

    st.write(f"F1 Score: {f1}")

    # Plotting the accuracy and F1 score
    metrics = ['Accuracy', 'F1 Score']
    scores = [accuracy, f1]

    fig, ax = plt.subplots()
    ax.barh(metrics, scores, color=['blue', 'orange'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_title('Chatbot Performance Metrics')

    st.pyplot(fig)
