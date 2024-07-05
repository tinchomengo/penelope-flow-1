# Penelope

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import dotenv
import os
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("\nLoading...")

# Authenticate and create the PyDrive client using a service account
def authenticate_google_drive():
    scope = ['https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)
    return drive

# Fetch PDF files from a specified Google Drive folder
def fetch_pdfs_from_drive_folder(drive, folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and mimeType contains 'application/pdf'"}).GetList()
    pdf_files = []
    for file in file_list:
        file_path = os.path.join("./documents", file['title'])
        file.GetContentFile(file_path)
        pdf_files.append(file_path)
    return pdf_files

# Parsing a local PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Splitting text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Vectorizing chunks with HuggingFaceEmbeddings
def vectorize_chunks(chunks):
    # Use HuggingFaceEmbeddings instead of OpenAIEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Creates a chain that combines the documents and the QA chat model
def create_custom_retrieval_chain(vectorstore):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

# Returns the answer to a question
def interact_with_llm(chain, query):
    result = chain.invoke({"input": query})
    return result['answer']

# Authenticate Google Drive
drive = authenticate_google_drive()

folder_id = os.getenv("FOLDER_ID")

# Fetch PDF files from the specified folder
pdf_files = fetch_pdfs_from_drive_folder(drive, folder_id)

# Process each PDF file
all_text = ""
for pdf_file in pdf_files:
    all_text += extract_text_from_pdf(pdf_file)

#all_text = "./documents/ThehistoryofBitcoinpdf.pdf"
#text = extract_text_from_pdf(all_text)
chunks = split_text_into_chunks(all_text)
vectorstore = vectorize_chunks(chunks)
chain = create_custom_retrieval_chain(vectorstore)


flag = False

while True:
    if flag == False:
        flag = True
        os.system('clear')
    query = input("Enter a question (type 'exit' to leave): ")
    if query == "exit":
        break
    answer = interact_with_llm(chain, query)
    print(answer)

