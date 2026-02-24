import hashlib
from pypdf import PdfReader
import io
import chromadb
from openai import OpenAI
import os

PERSIST_DIR = "chroma"
COLLECTION_NAME = "resume_rag"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

def extract_text_from_pdf(path: str):
  with open(path, "rb") as f:
    file_bytes = f.read()
  
  reader = PdfReader(io.BytesIO(file_bytes))
  pages = []
  for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    text = text.strip()
    if text:
      pages.append((i, text))
  return pages

def chunk_words(text: str, chunk_size: int = 800, overlap: int=150)->list[str]:
  words = text.split()
  if not words:
    return []
  
  if chunk_size<=0:
    raise ValueError("chunk_size must be >0")
  if overlap<0:
    raise ValueError("overlap must be >= 0")
  if overlap>=chunk_size:
    raise ValueError("overlap must be < chunk_size")
  
  chunks = []
  step = chunk_size - overlap

  for start in range(0, len(words), step):
    chunk = " ".join(words[start:start+chunk_size]).strip()
    if chunk:
      chunks.append(chunk)
  return chunks

def stable_id(text: str, source: str, page: int, chunk_index: int)->str:
  h = hashlib.sha1()
  h.update(text.encode("utf-8"))
  h.update(source.encode("utf-8"))
  h.update(str(page).encode("utf-8"))
  h.update(str(chunk_index).encode("utf-8"))
  return h.hexdigest()

def get_collection():
  client = chromadb.PersistentClient(path=PERSIST_DIR)
  collection = client.get_or_create_collection(COLLECTION_NAME)
  return collection

def embed_texts(texts: list[str], model: str = EMBED_MODEL)->list[list[float]]:
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")
  client = OpenAI(api_key=api_key)
  response = client.embeddings.create(
    model=model,
    input=texts,
  )
  return [item.embedding for item in response.data]

def add_to_index(pdf_paths: list[str], chunk_size: int = 800, overlap: int = 150) -> int:
  collection = get_collection()

  existing = collection.get(include=[])
  existing_ids = set(existing["ids"])

  texts_to_add = []
  ids_to_add = []
  metadatas_to_add = []

  for pdf_path in pdf_paths:
    pages = extract_text_from_pdf(pdf_path)

    for page_num, page_text in pages:
      chunks = chunk_words(page_text, chunk_size=chunk_size, overlap=overlap)

      for chunk_idx, chunk in enumerate(chunks, start=1):
        doc_id = stable_id(
          text=chunk,
          source=pdf_path,
          page=page_num,
          chunk_index=chunk_idx,
        )

        if doc_id in existing_ids: continue

        existing_ids.add(doc_id)
        texts_to_add.append(chunk)
        ids_to_add.append(doc_id)
        metadatas_to_add.append(
          {
            "source": pdf_path,
            "page": page_num,
            "chunk": chunk_idx,
          }
        )
  if not texts_to_add: return 0

  embeddings = embed_texts(texts_to_add)

  collection.add(
    ids=ids_to_add,
    documents=texts_to_add,
    embeddings=embeddings,
    metadatas=metadatas_to_add
  )
  return len(texts_to_add)