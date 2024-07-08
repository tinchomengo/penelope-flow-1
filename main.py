import os
import json
import pdfplumber
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Loading...")

# Authenticate and create the Google Drive client using a service account
def authenticate_google_drive():
    scope = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/documents.readonly']
    creds = Credentials.from_service_account_file('service_account.json', scopes=scope)
    drive_service = build('drive', 'v3', credentials=creds)
    docs_service = build('docs', 'v1', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)
    return drive_service, docs_service, sheets_service

# Fetch files from a specified Google Drive folder
def fetch_files_from_drive_folder(drive_service, folder_id):
    query = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.google-apps.spreadsheet')"
    results = drive_service.files().list(q=query).execute()
    return results.get('files', [])

# Fetch content from Google Docs
def fetch_google_doc_content(docs_service, file_id):
    document = docs_service.documents().get(documentId=file_id).execute()
    content = document.get('body').get('content')
    paragraph_text = ""
    table_matrix = []
    for element in content:
        if 'table' in element:
            for row in element.get('table').get('tableRows'):
                row_text = []
                for cell in row.get('tableCells'):
                    cell_text = ""
                    for text_run in cell.get('content'):
                        if 'paragraph' in text_run:
                            for elem in text_run.get('paragraph').get('elements'):
                                cell_text += elem.get('textRun').get('content').strip()
                    row_text.append(cell_text)
                table_matrix.append(row_text)
        elif 'paragraph' in element:
            for elem in element.get('paragraph').get('elements'):
                paragraph_text += elem.get('textRun').get('content').strip()

    return table_matrix, paragraph_text

# Fetch content from Google Sheets
def fetch_google_sheet_content(sheets_service, file_id):
    try:
        # Get sheet metadata
        sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=file_id).execute()
        sheet_names = [sheet['properties']['title'] for sheet in sheet_metadata['sheets']]
        
        all_text = []
        for sheet_name in sheet_names:
            result = sheets_service.spreadsheets().values().get(spreadsheetId=file_id, range=sheet_name).execute()
            values = result.get('values', [])
            all_text.extend(values)
        
        return all_text
    except Exception as e:
        print(f"An error occurred while fetching Google Sheets content: {e}")
        return []

# Fetch content from PDF files
def fetch_pdf_content(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Download a file from Google Drive
def download_file(drive_service, file_id, file_path):
    request = drive_service.files().get_media(fileId=file_id)
    with open(file_path, 'wb') as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return file_path

# Splitting text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Vectorizing chunks with HuggingFaceEmbeddings
def vectorize_chunks(chunks):
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
drive_service, docs_service, sheets_service = authenticate_google_drive()

folder_id = os.getenv("FOLDER_ID")

# Fetch files from the specified folder
files = fetch_files_from_drive_folder(drive_service, folder_id)

# Process each file
all_text = []
paragraphs = ""
for file in files:
    file_id = file['id']
    file_name = file['name']
    file_mime_type = file['mimeType']
    
    if file_mime_type == 'application/pdf':
        file_path = f"./documents/{file_name}"
        download_file(drive_service, file_id, file_path)
        all_text.append([fetch_pdf_content(file_path)])
    elif file_mime_type == 'application/vnd.google-apps.document':
        table_matrix, paragraph_text = fetch_google_doc_content(docs_service, file_id)
        all_text.extend(table_matrix)
        paragraphs += paragraph_text
    elif file_mime_type == 'application/vnd.google-apps.spreadsheet':
        all_text.extend(fetch_google_sheet_content(sheets_service, file_id))

# Convert the matrix into a string for vectorization
flat_text = '\n'.join([','.join(row) for row in all_text]) + "\n" + paragraphs

chunks = split_text_into_chunks(flat_text)
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