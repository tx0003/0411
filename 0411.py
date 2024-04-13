import os
import  openai
import PyPDF2

from langchain_openai import ChatOpenAI
 
chat=ChatOpenAI(
     openai_api_key=os.getenv("OPENAI_API_KEY"),
     model='gpt-3.5-turbo'
 )

def merge_pdf_text(input_path):
    with open(input_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        merged_text = ""
        for page in pdf_reader.pages:
            merged_text += page.extract_text()
    return merged_text

input_path = '民法典.pdf'
merged_text = merge_pdf_text(input_path)

with open("民法典.txt", "w", encoding="utf-8") as file:
    file.write(merged_text)

from langchain_community.document_loaders import TextLoader
 
txtloader = TextLoader("民法典.txt",encoding="utf-8")
 
txtdocument = txtloader.load_and_split()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n","。","\n",""],
    keep_separator=True,
)
docs=text_splitter.split_documents(txtdocument)


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

persist_directory='db'
embed_model=OpenAIEmbeddings()

#这里把向量存到db文件夹中了，然后就可以直接调用本地的向量库了
vectordb=Chroma.from_documents(documents=docs,embedding=embed_model,persist_directory=persist_directory) 
vectordb.persist()

#调用本地的向量库db
vectordb_load=Chroma(persist_directory=persist_directory,embedding_function=embed_model)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"。 
        以下是语料：
<context>
{context}
</context>

Question: {input}""")

#创建检索链
retriever = vectordb_load.as_retriever()
document_chain = create_stuff_documents_chain(chat, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain

memory = ConversationBufferMemory(llm=chat,memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=memory)
       

questions = [
  "小王在签合同时没认真看格式条款，对方也未做出说明，事后小王觉得自己遭遇“霸王条款”，相关条款有效吗？",
  "他后续应该怎么办？"
]
for question in questions:
        print(question)
        answer = qa.invoke(question)["answer"]
        print(answer)
        print()




