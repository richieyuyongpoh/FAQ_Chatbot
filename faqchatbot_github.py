import openai 
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_chat import message
import tiktoken

openai.api_key = st.secrets["api_secret"]


EMBEDDING_MODEL = "text-embedding-ada-002" 
COMPLETIONS_MODEL = "gpt-3.5-turbo"



qna_template = pd.read_csv("QnA_CompleteTemplate.csv")


qna_template = qna_template.set_index(["Item","Question"])




def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.Answer,EMBEDDING_MODEL) for idx, r in df.iterrows()
    }



document_embeddings = compute_doc_embeddings(qna_template)





def vector_similarity(x, y) :
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query,EMBEDDING_MODEL)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities




MAX_SECTION_LEN = 500
SEPARATOR = "\n* "


encoding = tiktoken.encoding_for_model(COMPLETIONS_MODEL)
separator_len = len(encoding.encode(SEPARATOR))



def construct_prompt(question, context_embeddings, df) :
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.Token + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.Answer.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
#     print(f"Selected {len(chosen_sections)} questions:")
#     print("\n".join(chosen_sections_indexes))
    
    header = """You are always Jane, Dr Yong Poh's assistant. Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "No MySejahtera FAQ is related to your question. Please try other questions."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query,
    df,
    document_embeddings,
    show_prompt = False
):
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    
    response = openai.ChatCompletion.create(
            model= COMPLETIONS_MODEL,
            temperature= 0.1,
            max_tokens= 300,
            messages=[{"role": "user", "content": prompt}])

    return response['choices'][0]["message"]["content"].strip(" \n")

             
             
st.title("FAQ Chatbot")


readme = st.checkbox("readme first")

if readme:

    st.write("""
        This is a customized FAQ Chatbot demo based on [FAQ MySejahtera](https://mysejahtera.malaysia.gov.my/faq_en/) using [ChatGPT API](https://openai.com/). 
        The web app is hosted on [streamlit cloud](https://streamlit.io/cloud).
        
        """)
    st.write ("For more info, please contact:")
    st.write("<a href='https://www.linkedin.com/in/yong-poh-yu/'>Dr. Yong Poh Yu </a>", unsafe_allow_html=True)
    
st.write("Instruction:")
st.write("")

st.write("Type the question in the following textbot. The AI assistant, Jane will assist you accordingly.")



# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
             
def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text
             
user_input = get_text()

if user_input:
    output = answer_query_with_context(user_input, qna_template, document_embeddings)    
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
             
if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
