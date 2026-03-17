import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF
PDF_PATH = "./assets/Grade9GTjoyluckclub.pdf"
DB_LOCATION = "./chrome_langchain_db"
EMBEDDING_MODEL = "mxbai-embed-large"


def load_and_split_pdf(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split the PDF in to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Define the size of each chunk
        chunk_overlap=200,  # Define the overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", " "],
    )

    # Preserve page numbers
    final_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            final_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **page.metadata
                    },  # page.metadata already contains the 'page' number
                )
            )

    # chunks = text_splitter.split_documents(documents)
    return final_chunks


def build_page_filter(start_page: int | None = None, end_page: int | None = None):
    if start_page is None and end_page is None:
        return None

    if start_page is None:
        start_page = end_page
    if end_page is None:
        end_page = start_page
    if start_page is None or end_page is None:
        return None
    if start_page < 1 or end_page < 1:
        raise ValueError("Page numbers must start at 1.")
    if start_page > end_page:
        raise ValueError("start_page must be less than or equal to end_page.")

    return {
        "$and": [
            {"page": {"$gte": start_page - 1}},
            {"page": {"$lte": end_page - 1}},
        ]
    }


def get_vector_store() -> Chroma:
    # Generate embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    is_add_documents = not os.path.exists(DB_LOCATION)  # check if already exists.
    has_existing_db = not is_add_documents and any(os.scandir(DB_LOCATION))

    if has_existing_db:
        return Chroma(
            persist_directory=DB_LOCATION,
            embedding_function=embeddings,
        )

    if not os.path.exists(PDF_PATH):
        print(
            f"Warning: PDF file not found at '{PDF_PATH}'. Starting with empty vector DB."
        )
        return Chroma(
            persist_directory=DB_LOCATION,
            embedding_function=embeddings,
        )

    final_chunks = load_and_split_pdf(PDF_PATH)
    # this vector db needs a separate list of ids as well
    return Chroma.from_documents(
        #    collection_name="restaurant_reviews",
        documents=final_chunks,
        persist_directory=DB_LOCATION,
        embedding=embeddings,
    )


vector_store = None


def search_documents(
    query: str,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
    k: int = 5,
):
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()

    page_filter = build_page_filter(start_page, end_page)
    search_kwargs = {"k": k}
    if page_filter is not None:
        search_kwargs["filter"] = page_filter

    return vector_store.similarity_search(query, **search_kwargs)


def get_retriever(k: int = 5):
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
