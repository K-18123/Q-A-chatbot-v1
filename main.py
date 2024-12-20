
import os
from operator import itemgetter
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Kết nối db
connection_string = f"mssql+pyodbc://khang:123456@localhost/Demo?driver=ODBC+Driver+17+for+SQL+Server"
db = SQLDatabase.from_uri(connection_string)

# Cấu hình LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#Dynamic Few-Shot Example Selection
examples = [
    {
        "input": "Chi tiết danh sách khách hàng?",
        "query": "SELECT * FROM custome"
    },
    {
        "input": "Danh sách khách hàng?",
        "query": "SELECT [customer_name] FROM customer"
    }
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
     examples,
     OpenAIEmbeddings(),
     FAISS,
     k=1,
     input_keys=["input"],
 )

few_shot_prompt = FewShotChatMessagePromptTemplate(
     example_prompt=example_prompt,
     example_selector=example_selector,
     input_variables=["input","top_k"],
 )


#Dynamic Relevant Table Selection
def get_table_details():
    table_description = pd.read_csv("data_table_description.csv")
    table_docs = []

    # Duyệt qua các hàng của DataFrame để tạo đối tượng Document
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

table_details = get_table_details()\

table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:
{table_details}
Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

# table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)

def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables

select_table = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables

#memory
history = ChatMessageHistory()

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         "Bạn là một chuyên gia MySQL. Dựa trên câu hỏi đầu vào, hãy tạo một truy vấn MySQL cú pháp chính xác để thực thi. Trừ khi có yêu cầu cụ thể khác.
         Giữ nguyên các giá trị được truy vấn trong SQL
         Dưới đây là thông tin về bảng liên quan: {table_info}
         Dưới đây là một số ví dụ về câu hỏi và truy vấn SQL tương ứng.
         """
         ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)
#Tạo một chain -> SQL query với final_prompt
generate_query = create_sql_query_chain(llm, db, final_prompt)
#truy vấn SQl trên csdl
execute_query = QuerySQLDatabaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
)
rephrase_answer = answer_prompt | llm | StrOutputParser()
#Main chain
chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
)

#streamlit

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hiển thị lịch sử trò chuyện
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhập câu hỏi từ người dùng
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Lưu câu hỏi của người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Hiển thị câu hỏi của người dùng
    with st.chat_message("user"):
        st.markdown(user_input)

    # Xử lý câu hỏi và trả lời
    with st.chat_message("assistant"):
        try:
            # Gọi chain để xử lý câu hỏi
            response = chain.invoke({"question": user_input, "messages": history.messages})
            assistant_response = response  # Kết quả trả lời
        except Exception as e:
            assistant_response = f"Lỗi: {str(e)}"

        # Hiển thị câu trả lời
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})


#Streamlit run main.py