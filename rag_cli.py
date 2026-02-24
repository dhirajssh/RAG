import hashlib
from pypdf import PdfReader
import io
import chromadb

PERSIST_DIR = "chroma"
COLLECTION_NAME = "resume_rag"

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