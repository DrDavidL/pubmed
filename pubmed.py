import json
import time
from openai import OpenAI
from prompts import search_term_system_prompt
import os
import streamlit as st
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import requests
import time
from retrying import retry

import os 
from io import StringIO, BytesIO
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import streamlit as st
import pdfplumber
# from PyMuPDFLoader import fitz

import fitz
import os
from io import StringIO



st.set_page_config(page_title='PubMed Researcher', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("PubMed Researcher")
st.write("ALPHA version 0.5")


with st.expander('About PubMed Researcher - Important Disclaimer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.info(disclaimer)
    st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.5, 0.01)
    st.write("Last updated 6/20/24")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


@st.cache_data
def create_chat_completion(
    messages,
    model="gpt-4o",
    frequency_penalty=0,
    logit_bias=None,
    logprobs=False,
    top_logprobs=None,
    max_tokens=None,
    n=1,
    presence_penalty=0,
    response_format=None,
    seed=None,
    stop=None,
    stream=False,
    include_usage=False,
    temperature=1,
    top_p=1,
    tools=None,
    tool_choice="none",
    user=None
):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Prepare the parameters for the API call
    params = {
        "model": model,
        "messages": messages,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "max_tokens": max_tokens,
        "n": n,
        "presence_penalty": presence_penalty,
        "response_format": response_format,
        "seed": seed,
        "stop": stop,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "user": user
    }

    # Handle the include_usage option for streaming
    if stream:
        params["stream_options"] = {"include_usage": include_usage}
    else:
        params.pop("stream_options", None)

    # Handle tools and tool_choice properly
    if tools:
        params["tools"] = [{"type": "function", "function": tool} for tool in tools]
        params["tool_choice"] = tool_choice

    # Handle response_format properly
    if response_format == "json_object":
        params["response_format"] = {"type": "json_object"}
    elif response_format == "text":
        params["response_format"] = {"type": "text"}
    else:
        params.pop("response_format", None)

    # Remove keys with None values
    params = {k: v for k, v in params.items() if v != None}
    
    completion = client.chat.completions.create(**params)
    
    return completion

@st.cache_data
def display_articles_with_streamlit(articles):
    i = 1
    for article in articles:
        st.write(f"{i}. {article['title']}[{article['year']}]({article['link']})")
        i+=1
        # st.write("---")  # Adds a horizontal line for separation


# Example usage:
# display_articles_with_streamlit(articles)

def set_llm_chat(model, temperature):
    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        return ChatOpenAI(model=model, openai_api_base = "https://api.openai.com/v1/", openai_api_key = st.secrets["OPENAI_API_KEY"], temperature=temperature)
    else:
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
          "X-Title": "GPT and Med Ed"}
        return ChatOpenAI(model = model, openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = st.secrets["OPENROUTER_API_KEY"], temperature=temperature, max_tokens = 500, headers=headers)


@st.cache_data
def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context, model, print = True):

    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    if history_context == None:
        history_context = ""
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        openai.api_base = "https://api.openai.com/v1/"
        openai.api_key = st.secrets['OPENAI_API_KEY']
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = 500,
        stream = True,   
        )
    else:      
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = st.secrets["OPENROUTER_API_KEY"]
        # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        route = "fallback",
        messages = messages,
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
            "X-Title": "GPT and Med Ed"},
        temperature = temperature,
        max_tokens = 500,
        stream = True,   
        )
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:   
        if print:     
            c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content





@retry(
    retry_on_exception=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429,
    stop_max_attempt_number=5,  # Maximum number of retry attempts
    wait_exponential_multiplier=1000,  # Initial wait time in milliseconds
    wait_exponential_max=10000  # Maximum wait time in milliseconds
)
def make_request(url):
    response = requests.get(url)
    response.raise_for_status()
    return response




@st.cache_data
def pubmed_abstracts_new(search_terms, search_type="all"):
    # URL encoding
    search_terms_encoded = requests.utils.quote(search_terms)

    # Define the publication type filter based on the search_type parameter
    if search_type == "all":
        publication_type_filter = ""
    elif search_type == "clinical trials":
        publication_type_filter = "+AND+Clinical+Trial[Publication+Type]"
    elif search_type == "reviews":
        publication_type_filter = "+AND+Review[Publication+Type]"
    else:
        raise ValueError("Invalid search_type parameter. Use 'all', 'clinical trials', or 'reviews'.")

    # Construct the search query with the publication type filter
    search_query = f"{search_terms_encoded}{publication_type_filter}"
    
    # Query to get the top 20 results
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax=20"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Check if no results were returned
        if 'count' in data['esearchresult'] and int(data['esearchresult']['count']) == 0:
            return [], ""  # No results found
    except Exception as e:
        st.error(f"Error fetching search results: {e}")
        return [], ""

    ids = data['esearchresult']['idlist']
    articles = []

    for id in ids:
        details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json"
        try:
            response = requests.get(details_url)
            response.raise_for_status()
            details = response.json()
            if 'result' in details and str(id) in details['result']:
                article = details['result'][str(id)]
                year = article['pubdate'].split(" ")[0]
                if year.isdigit():
                    articles.append({
                        'title': article['title'],
                        'year': year,
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{id}"
                    })
            else:
                st.warning(f"Details not available for ID {id}")
        except Exception as e:
            st.error(f"Error fetching details for ID {id}: {e}")

    # Second query: Get the abstract texts for the top 10 results
    abstracts = []
    for id in ids:
        abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id}&retmode=text&rettype=abstract"
        try:
            response = requests.get(abstract_url)
            response.raise_for_status()
            abstract_text = response.text
            if "API rate limit exceeded" not in abstract_text:
                abstracts.append(abstract_text)
        except Exception as e:
            st.error(f"Error fetching abstract for ID {id}: {e}")

    return articles, "\n".join(abstracts)






# Longer approach to handle cases with no results
def pubmed_abstracts(search_terms, search_type="all"):
    # URL encoding
    search_terms_encoded = requests.utils.quote(search_terms)

    # Define the publication type filter based on the search_type parameter
    if search_type == "all":
        publication_type_filter = ""
    elif search_type == "clinical trials":
        publication_type_filter = "+AND+Clinical+Trial[Publication+Type]"
    elif search_type == "reviews":
        publication_type_filter = "+AND+Review[Publication+Type]"
    else:
        raise ValueError("Invalid search_type parameter. Use 'all', 'clinical trials', or 'reviews'.")

    # Construct the search query with the publication type filter
    search_query = f"{search_terms_encoded}{publication_type_filter}"
    
    # Query to get the top 20 results
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax=20&api_key={st.secrets['pubmed_api_key']}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Check if no results were returned, and if so, use a longer approach
        if 'count' in data['esearchresult'] and int(data['esearchresult']['count']) == 0:
            return st.write("No results found. Try a different search or try again after re-loading the page.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching search results: {e}")
        return []

    ids = data['esearchresult']['idlist']
    articles = []

    for id in ids:
        details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json&api_key={st.secrets['pubmed_api_key']}"
        try:
            details_response = requests.get(details_url)
            details_response.raise_for_status()  # Raise an exception for HTTP errors
            details = details_response.json()
            if 'result' in details and str(id) in details['result']:
                article = details['result'][str(id)]
                year = article['pubdate'].split(" ")[0]
                if year.isdigit():
                    articles.append({
                        'title': article['title'],
                        'year': year,
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{id}"
                    })
            else:
                st.warning(f"Details not available for ID {id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching details for ID {id}: {e}")
        time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    # Second query: Get the abstract texts for the top 10 results
    abstracts = []
    for id in ids:
        abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id}&retmode=text&rettype=abstract&api_key={st.secrets['pubmed_api_key']}"
        try:
            abstract_response = requests.get(abstract_url)
            abstract_response.raise_for_status()  # Raise an exception for HTTP errors
            abstract_text = abstract_response.text
            if "API rate limit exceeded" not in abstract_text:
                abstracts.append(abstract_text)
            else:
                st.warning(f"Rate limit exceeded when fetching abstract for ID {id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching abstract for ID {id}: {e}")
        time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    return articles, "\n".join(abstracts)



@st.cache_data
def web_search(search_query):
    # Artificial sleep so the UI is clearer
    time.sleep(1.5)
    search = DuckDuckGoSearchAPIWrapper()
    return search.run(search_query)


@st.cache_data
def pubmed_search(search_terms):
    # URL encoding
    search_terms_encoded = requests.utils.quote(search_terms)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_terms_encoded}&retmode=json"

    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"Error fetching search results: {e}")
        return []

    ids = data['esearchresult']['idlist']
    articles = []

    for id in ids:
        details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json"
        try:
            details_response = requests.get(details_url)
            details = details_response.json()
        except Exception as e:
            print(f"Error fetching details for ID {id}: {e}")
            continue

        # Check if 'result' key and the specific PubMed ID exist in the details
        if 'result' in details and str(id) in details['result']:
            article = details['result'][str(id)]
            if 'title' in article and 'pubdate' in article:
                year = article['pubdate'].split(" ")[0]
                if year.isdigit():  # Check if the extracted year is a number
                    articles.append({
                        'title': article['title'],
                        'year': year,
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{id}"
                    })
    # Sort articles in reverse chronological order
    articles.sort(key=lambda x: x['year'], reverse=True)

    return articles

def split_texts(text, chunk_size, overlap, split_method):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        # st.error("Failed to split document")
        st.stop()

    return splits


    



@st.cache_data
def create_retriever(texts, name, save_vectorstore=False):
    
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",
                                  openai_api_base = "https://api.openai.com/v1/",
                                  openai_api_key = st.secrets['OPENAI_API_KEY']
                                  )
    try:
        vectorstore = FAISS.from_texts(texts, embeddings)
        if save_vectorstore:
            vectorstore.save_local(f"{name}.faiss")
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever



if "abstract_questions" not in st.session_state:
    st.session_state["abstract_questions"] = []
    
if "abstract_answers" not in st.session_state:
    st.session_state["abstract_answers"] = []

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = []

if "abstracts" not in st.session_state:
    st.session_state["abstracts"] = ""

if "temp" not in st.session_state:
    st.session_state["temp"] = 0.5
    
if "your_question" not in st.session_state:
    st.session_state["your_question"] = ""
    
if "texts" not in st.session_state:
    st.session_state["texts"] = ""
    
if "retriever" not in st.session_state:
    st.session_state["retriever"] = ""
    
if "citations" not in st.session_state:
    st.session_state["citations"] = ""
    
if "search_terms" not in st.session_state:
    st.session_state["search_terms"] = ""   

if check_password():

    st.session_state.model = st.sidebar.selectbox("Model Options", ("openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-16k", "openai/gpt-4", "anthropic/claude-instant-v1", "google/palm-2-chat-bison", "meta-llama/llama-2-70b-chat", "gryphe/mythomax-L2-13b", "nousresearch/nous-hermes-llama2-13b"), index=1)

    st.warning("""This tool performs PubMed searches and then analyzes available abstracts in order to answer your question. Clearly,
            this will often be inadequate and is intended to illustrate an AI approach that will become (much) better with time. View citations and abtracts on the left sidebar. """)

    
    option = st.sidebar.radio("Select an Option", ("Ask a Question", "Ask about a PDF", "Ask about the Abstracts"))
    
    your_question = st.text_input("Your question for PubMed", placeholder="Enter your question here")
    st.session_state.your_question = your_question
    search_type = st.radio("Select an Option", ("all", "clinical trials", "reviews"), horizontal=True)




    if st.session_state.your_question != "":
        # search_terms = answer_using_prefix("Convert the user's question into relevant PubMed search terms; include related MeSH terms to improve sensitivity.", 
                                    # "What are the effects of intermittent fasting on weight loss in adults over 50?",
                                    # "(Intermittent fasting OR Fasting[MeSH]) AND (Weight loss OR Weight Reduction Programs[MeSH]) AND (Adults OR Middle Aged[MeSH]) AND Age 50+ ", st.session_state.your_question, 0.5, None, st.session_state.model, print= False)
        
        messages_search_terms = [{'role': 'system', 'content': search_term_system_prompt},
            {'role': 'user', 'content': st.session_state.your_question}],
        search_terms = create_chat_completion(messages=messages_search_terms)                            
        st.write(f'Here are your search terms: {search_terms}')      
                                     
        st.session_state.search_terms = search_terms
        with st.sidebar.expander("Current Question", expanded=False):
            st.write(st.session_state.your_question)
            st.write('Search terms used: ' + st.session_state.search_terms)
        with st.spinner("Searching PubMed... (Temperamental - ignore errors if otherwise working. API access can take a minute or two.)"):
            if st.session_state.search_terms != "":
                st.session_state.citations, st.session_state.abstracts = pubmed_abstracts(st.session_state.search_terms, search_type=search_type)
                if st.session_state.citations == [] or st.session_state.abstracts == "":
                    st.warning("The PubMed API is tempermental. Refresh and try again.")
                    st.stop()


        # st.write(st.session_state.citations)
        # st.write(st.session_state.abstracts)

    with st.sidebar.expander("Show citations"):
        display_articles_with_streamlit(st.session_state.citations)
    with st.sidebar.expander("Show abstracts"):
        st.write(st.session_state.abstracts)
    system_context_abstracts = """You receive user query terms and PubMed abstracts for those terms as  your inputs. You first provide a composite summary of all the abstracts emphasizing any of their conclusions. Next,
    you provide key points from the abstracts in order address the user's likely question based on the on the query terms.       
    """

    # Unblock below if you'd like to submit the full abtracts. This is not recommended as it is likely to be too long for the model.

    # prompt_for_abstracts = f'User question: {your_question} Abstracts: {st.session_state.abstracts} /n/n Generate one summary covering all the abstracts and then list key points to address likely user questions.'

    # with st.spinner("Waiting on LLM analysis of abstracts..."):
    #     full_answer = answer_using_prefix(system_context_abstracts, "","",prompt_for_abstracts, 0.5, None, st.session_state.model)
    # with st.expander("Show summary and key points"):
    #     st.write(f'Here is the full abstracts inferred answers: {full_answer}')


    # st.write("'Reading' all the abstracts to answer your question. This may take a few minutes.")


    st.info("""Next, words in the abstracts are converted to numbers for analysis. This is called embedding and is performed using an OpenAI [embedding model](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and then indexed for searching. Lastly,
            your selected model (e.g., gpt-3.5-turbo-16k) is used to answer your question.""")


    if st.session_state.abstracts != "":
        
        # with st.spinner("Embedding text from the abstracts (lots of math can take a couple minutes to change words into vectors)..."):
        #     st.session_state.retriever = create_retriever(st.session_state.abstracts)

        with st.spinner("Splitting text from the abstracts into concept chunks..."):
            st.session_state.texts = split_texts(st.session_state.abstracts, chunk_size=1250,
                                        overlap=200, split_method="splitter_type")
        with st.spinner("Embedding the text (converting words to vectors) and indexing to answer questions about the abtracts (Takes a couple minutes)."):
            st.session_state.retriever = create_retriever(st.session_state.texts)


        # openai.api_base = "https://openrouter.ai/api/v1"
        # openai.api_key = st.secrets["OPENROUTER_API_KEY"]

        llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
        # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_base = "https://api.openai.com/v1/")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.retriever)

    else:
        st.warning("No files uploaded.")       
        st.write("Ready to answer your questions!")


    abstract_chat_option = st.radio("Select an Option", ("Ask More Questions", "Summary"))
    if abstract_chat_option == "Summary":
        user_question_abstract = "Summary: Using context provided, generate a concise and comprehensive summary. Key Points: Generate a list of Key Points by using a conclusion section if present and the full context otherwise."
    if abstract_chat_option == "Ask More Questions":
        user_question_abstract = st.text_input("If desired modify and ask additional questions of the retrieved abstracts. Here was your initial question:", your_question)

    index_context = f'Use only the reference document for knowledge. Question: {user_question_abstract}'
    if st.session_state.abstracts != "":
        abstract_answer = qa(index_context)

        if st.button("Ask more about the abstracts"):
            index_context = f'Use only the reference document for knowledge. Question: {user_question_abstract}'
            abstract_answer = qa(index_context)

        # Append the user question and PDF answer to the session state lists
        st.session_state.abstract_questions.append(user_question_abstract)
        st.session_state.abstract_answers.append(abstract_answer)

        # Display the PubMed answer
        st.write(abstract_answer["result"])

        # Prepare the download string for the PDF questions
        abstract_download_str = f"{disclaimer}\n\nPDF Questions and Answers:\n\n"
        for i in range(len(st.session_state.abstract_questions)):
            abstract_download_str += f"Question: {st.session_state.abstract_questions[i]}\n"
            abstract_download_str += f"Answer: {st.session_state.abstract_answers[i]['result']}\n\n"

        # Display the expander section with the full thread of questions and answers
        with st.expander("Your Conversation with your Abstracts", expanded=False):
            for i in range(len(st.session_state.abstract_questions)):
                st.info(f"Question: {st.session_state.abstract_questions[i]}", icon="🧐")
                st.success(f"Answer: {st.session_state.abstract_answers[i]['result']}", icon="🤖")

            if abstract_download_str:
                st.download_button('Download', abstract_download_str, key='abstract_questions_downloads')
        



                
