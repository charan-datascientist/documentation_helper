import os
from dotenv import load_dotenv
load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



def ingest_docs():
    loader = ReadTheDocsLoader("/Users/charankumarraghupatruni/workspace/documentation-helper/documentation-helper.git/langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap = 50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    PineconeVectorStore.from_documents(documents, embeddings, index_name = os.environ["INDEX_NAME"])




if __name__ == "__main__":
    ingest_docs()





