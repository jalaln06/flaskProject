from langchain.chains import AnalyzeDocumentChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
import os

os.environ['OPENAI_API_KEY'] = 'INSERT KEY HERE'
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
qa_chain = load_qa_chain(llm, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
# def init_model():
#
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
#     qa_chain = load_qa_chain(llm, chain_type="map_reduce")
#     qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

def run_model_on_text(text):

    return qa_document_chain.run(input_document=text,
                          question="Imagine you're my research assistant in the sphere of biology. I'm trying to understand which papers shall I read firsthand in order to understand research faster. What will I get as knowledge if I read this paper?")
