from pypdf import PdfReader
import io

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