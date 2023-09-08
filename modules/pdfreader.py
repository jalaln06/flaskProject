from langchain.document_loaders import PyPDFLoader
import urllib.request
import io
import PyPDF2
def read_pdf():
    loader = PyPDFLoader("/Users/jalal/PycharmProjects/flaskProject/static/content/2309.00408.pdf")
    pages = loader.load_and_split()
    text = ""
    for page in pages:
        text += page.page_content
    return text

def read_pdf_from_url(url = 'https://arxiv.org/pdf/2308.15311.pdf'):
    
    URL = url
    req = urllib.request.Request(URL, headers={'User-Agent': "Magic Browser"})
    remote_file = urllib.request.urlopen(req).read()
    remote_file_bytes = io.BytesIO(remote_file)
    pdfdoc_remote = PyPDF2.PdfReader(remote_file_bytes)
    text = ''
    for i in range(len(pdfdoc_remote.pages)):
        page = pdfdoc_remote.pages[i]
        text += page.extract_text()
    return text
