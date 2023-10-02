"""
Get Started
"""

import os
import getpass

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


ROOT_DIR = './data-diff'
USERNAME = 'sungchun12'

embeddings = OpenAIEmbeddings(disallowed_special=())

def get_openai_api_key() -> None:
    if 'OPENAI_API_KEY' in os.environ:
        open_ai_key = os.environ['OPENAI_API_KEY']
    else:
        open_ai_key = getpass.getpass('Enter your OpenAI API key: ')
        os.environ['OPENAI_API_KEY'] = open_ai_key

def load_docs(root_dir) -> list:
    docs = []

    for dirpath, dirnames, filesnames in os.walk(root_dir):
        for filename in filesnames:
            try:
                loader = TextLoader(os.path.join(dirpath, filename), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(e)

    print(f'Loaded {len(docs)} documents from {root_dir}')
    return docs

def split_text_to_chunks(docs) -> None:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    return chunks

def load_embeddings(username, chunks) -> None:
    db = DeepLake(dataset_path=f"hub://{username}/data-diff-chat", embedding=embeddings, public=True)
    db.add_documents(chunks)

def access_vectorstore(username) -> None:
    db = DeepLake(dataset_path=f"hub://{username}/data-diff-chat", embedding=embeddings, read_only=True)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    return retriever

def setup_chatbot(retriever) -> str:
    model = ChatOpenAI(model='gpt-4')
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    return qa

questions = ["What unit testing framework does data-diff use within the codebase ?"]

def chat_demo(questions, qa) -> str:
    chat_history = []
    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"-> **Answer**: {result['answer']} \n")


def main():
    get_openai_api_key()
    # docs = load_docs(ROOT_DIR)
    # chunks = split_text_to_chunks(docs)
    # load_embeddings(USERNAME, chunks)
    retriever = access_vectorstore(USERNAME)
    qa = setup_chatbot(retriever)
    chat_demo(questions, qa)


if __name__ == '__main__':
    main()