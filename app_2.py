import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import streamlit as st
import pandas as pd
# import plotly.express as px
from streamlit_option_menu import option_menu
local = True
import streamlit.components.v1 as components


import streamlit as st
import torch
import time
from sentence_transformers import SentenceTransformer, util
from streamlit_option_menu import option_menu
import re
from io import StringIO

@st.cache_resource
def read_chunks():
  
    print("reading in chunks")
    
    
#     spark = SparkSession.builder.appName("Databricks Shell").getOrCreate()
    
    


    df = pd.read_csv("NvidiaDocumentationQandApairs.csv")

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    corpus = df["question"].tolist()
    answers = df["answer"].tolist()
    corpus_embedings = model.encode(corpus, convert_to_tensor=True)
    return model, df, corpus_embedings, corpus, answers

# from google.colab import drive
# drive.mount("/gdrive")

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''



# About Page

def about():
    st.title('About')
    
    st.write("""
    Write stuff here""")
    
    st.markdown("""
    Here's a brief overview of the data:

   """)
    

# Contact Us Information
def contact_us():
    st.title('Contact Us')
    st.markdown('''
    If you have any questions or suggestions regarding our project, feel free to reach out to any of our team members:

    - **Stavros Kontzias**: [stavros.kontzias@example.com](mailto:stavros.kontzias@example.com)
    ''')







def faq():
    st.markdown(
        """
# FAQ
## How does Smart Search work?
Each QA document was divided into smaller chunks 
and stored in a special type of database called a vector index 
that allows for semantic search and retrieval.
When you ask a question, the Smart Search tool will search through the
document chunks and find the most relevant ones using the vector index.
## Is my data safe?
Yes, your data is safe.
## Why does it take so long to index my document?
We currently do not have access to GPU hardware, therefore we are forced to use CPU
to compute which chunk is most similar to your question, and there’s a lot of chunks!
## Are the answers 100% accurate?
No, the answers are not 100% accurate. It uses the embeddings
of each chunk and the question to find the most similiar paragraph.
Semantic search finds the most relevant chunks and does not see the entire document,
which means that it may not be able to find all the relevant information and
may not be able to answer all questions (especially summary-type questions
or questions that require a lot of context from the document).
But for most use cases, Smart Search is accurate
Always check with the sources to make sure that the answers are correct.
"""
    )

def sidebar():
    with st.sidebar:

        with st.columns(3)[1]:
            if local:
                st.image('uva.png')
            else:
                st.image('/content/drive/My Drive/msds_bayes_project/uva.png')
#         st.image('./advana.png')


#         choose = option_menu("App Gallery", ["About", "Photo Editing", "Project Planning", "Python e-Course", "Contact"],
#                          icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
#                          menu_icon="app-indicator", default_index=0, orientation="horizontal",
#                          styles={
#         "container": {"padding": "5!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"}, 
#         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "#02ab21"},
#                 }
#             )
        st.markdown(
            "## How to use\n"
            "1. Navigate using Navigatio Menu💬\n"  # noqa: E501
            "2. Tweak posterior distribution, priors, etc.\n"
            "3. Get Response\n"
        )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖This app is our submission for the final project in DS 6050 "
            "and contains all the require materials for submission "
        )
        st.markdown(
            "This tool is a work in progress.\n"
            "Email your feedback and suggestions to [UVA Masters Students💡](mailto:skontzias@gmail.com) "  # noqa: E501"
        )
        st.markdown("Made by [UVA DS Team](https://uva.edu)")
        st.markdown("---")

        faq()


def split_text_into_chunks(text, parse_method):
    # Split essay into chunks based on paragraph markings inside parentheses
    if parse_method == "Portion Marking":
      chunks = re.split(r'(\(\w+\))', text)
    elif parse_method == "Paragraphs":
      chunks = re.split('\n\n', text)
    elif parse_method == "Sentences":
      
      import nltk.data

      _sent_detector = nltk.data.load('english.pickle')

      def split_sentence(text):
        # Split text.
        sentences = _sent_detector.tokenize(text)
        # Find each sentence's offset.
        sent_list = []
        needle = 0
        for sent in sentences:
            start = text.find(sent, needle)
            end = start + len(sent) - 1
            needle += len(sent)
            sent_list.append(sent)
        # Return results
        return sent_list
      chunks = split_sentence(text)
    elif parse_method == "Langchain":     
      from langchain.text_splitter import CharacterTextSplitter
      text_splitter = CharacterTextSplitter(chunk_size=256,chunk_overlap= 20)
      docs = text_splitter.create_documents([text])
      chunks = [doc.page_content for doc in docs]
    else:
      parse_method = "(" + str(parse_method) + ")"
      raw_s = r'{}'.format(parse_method)
      chunks = re.split(raw_s, text)
      combined_splits = []
      for i in range(1, len(chunks), 2):
          combined_splits.append(chunks[i] + chunks[i+1])
      chunks = combined_splits
      
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Combine chunks with less than 10 words with the next chunk
    combined_chunks = []
    current_chunk = ''
    for chunk in chunks:
        chunk_cleaned = chunk.replace('\n', ' ')
        current_chunk += ' ' + chunk_cleaned if current_chunk.count(' ') < 10 else ''
        if current_chunk.count(' ') >= 10:
            combined_chunks.append(current_chunk.strip())
            current_chunk = ''

    if current_chunk:
        combined_chunks.append(current_chunk.strip())

    # Create DataFrame with paragraph markings attached to chunk content
    data = {'chunk': combined_chunks}
    df = pd.DataFrame(data)

    return df





@st.cache_data
def embed_custom(_model, custom_df):          
      custom_corpus = custom_df["chunk"].tolist()
      custom_corpus_embedings = _model.encode(custom_corpus, convert_to_tensor=True)
      return custom_df, custom_corpus_embedings, custom_corpus





def make_request(model, df, corpus_embedings, corpus,answers, model_choice, question_input="Classification", k_chunks=1):

    
    #top_k = min(1, len(corpus))
    print("Converting question to embedings")
    query_embedding = model.encode(question_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embedings)[0]
    top_results = torch.topk(cos_scores, k=k_chunks)
#     print("\n\n========================\n\n")
#     print("Query:", query)
#     print("\n\n========================\n\n")
    if model_choice != "Embedings":
        resp_list = []
        for score, idx in zip(top_results[0], top_results[1]):
    #         resp_list.append(f"{corpus[idx]} | (Score {score:.2f})")
            resp_list.append(f"{answers[idx]}\n")
    #         ---
    else:
        
        resp_list = []
        for score, idx in zip(top_results[0], top_results[1]):
    #         resp_list.append(f"{corpus[idx]} | (Score {score:.2f})")
            resp_list.append(f"{answers[idx]}\n\n**Score:** {score:.2f}\n\n---\n")
    #         ---

# **Score:** 0.58  
# **Filename:** Classification Guidlines.pdf
      
#       st.session_state.chat_history.append(question_input)
#       st.session_state.chat_history.append(corpus[top_results[1]])
#       return f"{corpus[idx]} | (Score {score:.4f})"
    return "\n\n\n\n".join(resp_list)










def print_response(assistant_response):
             
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in assistant_response.split(" "):
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
# Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})













def chat(model, df, corpus_embedings, corpus, answers):
  st.header("📖Chat about NVIDIA GPU's")
    
    
#     question_input = False
  response = False
#     question_input = st.text_input("Ask a question about JCOFA Narrative Files")
  # query = st.text_input("Ask a question about JCOFA Narrative Files", on_change=clear_submit)
  with st.expander("Advanced Options"):
      text = ''
      col1, col2 = st.columns([2, 3])
      with col1:
  #         st.metric(label="Temp", value="273 K", delta="1.2 K")
          show_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
          show_full_doc = st.checkbox("Show parsed contents of the document")
          model_choice = st.selectbox("Model", ["Embedings", "gpt 3.5", "gpt 3.5 RAG Mode", "Fine-Tuned Custom GPT"],help="Do you want to use LLM or just Embedings Model" )
          k_chunks = st.selectbox("Chunks to Return", [1,2,3,4,5],help="How many chunks returned do you want for each query?" )
#               image_file = st.file_uploader("Upload PDF or Docx",type=['pdf', 'docx'])
#               if image_file is not None:
#                   file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
#   #                 st.write(file_details)
#                   if image_file.name.endswith("pdf"):
#                     import PyPDF2
#                     st.write("Extracting PDF")
#                     reader = PyPDF2.PdfFileReader(image_file)
#                     text = ""
#                     for page in reader.pages:
#                         text += page.extractText() + "\n"
#                     text_final = split_text_into_chunks(text)
#   #                   st.write(text)
#                   elif image_file.name.endswith("docx"):
#                     st.write("Extracting Word Doc")
#                     import docx
#                     doc = docx.Document(image_file)
#                     fullText = []
#                     for para in doc.paragraphs:
#                         fullText.append(para.text)
#                     text = '\n'.join(fullText)
#                     text_final = split_text_into_chunks(text)
#           with col2:
#               st.write(text)




#     submit = st.button("Submit")

#     st.markdown("""---""")


  ###################################

    # Initialize chat history
  if "messages" not in st.session_state:
      st.session_state.messages = []

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])

  # Accept user input
  if prompt := st.chat_input("Ask me about NVIDIA Products"):
      # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": prompt})
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(prompt)

      # Display assistant response in chat message container
      with st.chat_message("assistant"):
        assistant_response = make_request(model, df, corpus_embedings, corpus, answers, model_choice, prompt, k_chunks)
        if model_choice == "Embedings":
            print_response(assistant_response)
        elif model_choice == "gpt 3.5 RAG Mode":

            import openai
            openai.api_key = st.secrets["key"]

#             st.write(assistant_response)
            with st.spinner("waiting for API response"):
                completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                    {"role": "system", "content": f"You are a Computer Eningeer, knowledgable in computing hardware. Use the below text to answer the promt: {assistant_response}"},
                    {"role": "user", "content": prompt}
                  ]
                )
                print_response(completion.choices[0].message["content"])
        elif model_choice == "gpt 3.5":

            import openai
            openai.api_key = st.secrets["key"]
            with st.spinner("waiting for API response"):
                completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                  ]
                )

            print(completion.choices[0].message)
            print_response(completion.choices[0].message["content"])


            
        elif model_choice == "Fine-Tuned Custom GPT":     
            import openai
            openai.api_key = st.secrets["key"]
            message = []
            message.append({"role": "system", "content": "You are a python expert, proficient in  object orientated programming"})
            message.append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(
            model='ft:gpt-3.5-turbo-0613:personal:python-ds6050:8SZAw9k7', messages=message, temperature=0, max_tokens=500
            )
            print(response["choices"][0]["message"]["content"])
            print_response(response["choices"][0]["message"]["content"])




            
            
  if show_full_doc and doc:
      with st.expander("Document"):
          # Hack to get around st.markdown rendering LaTeX
          st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)    
          






def main():

    # clear_submit()
    st.set_page_config(page_title="UVA DS 6050 Final Project", page_icon="📖", layout="wide")
    st.write(css, unsafe_allow_html=True)
    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    model, df, corpus_embedings, corpus, answers = read_chunks()
    

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "download" not in st.session_state:
        st.session_state.download = None
    st.session_state.response = False
    if "menu_select" not in st.session_state:
        st.session_state.menu_select = 0
 

    
    
    
    
    
    
    

    st.header("📖UVA DS 6050 Final Project")
    st.markdown("""---""")
    sidebar()
    menu = option_menu(menu_title=None, options=["About/EDA", "Chat", "Model Dev", "Project Essay", "Contact Us"],
                         icons=['house', 'chat dots fill', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0, orientation="horizontal",
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#175A99"},
                }
            )
    
    
    if menu == "About":
        about()
        
    elif menu == "Model Dev":    
        pass
    
    elif menu == "Contact Us":
        contact_us()
      
    elif menu == "Custom PDF/Doc":
        custom_doc(model)
            
    elif menu == "Chat":
        chat(model, df, corpus_embedings, corpus, answers)

    elif menu == "EDA":
        pass


if __name__ == '__main__':
    main()
