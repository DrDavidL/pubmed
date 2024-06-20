import streamlit as st
from openai import OpenAI
from prompts import search_term_system_prompt
import requests
import time
import json

def json_to_markdown(json_data):
    """
    Converts a JSON object to a nicely formatted markdown string.

    Parameters:
    json_data (dict): The JSON object containing the data to be converted.

    Returns:
    str: A markdown formatted string.
    """
    markdown_str = ""
    i = 1
    for item in json_data:
        title = item.get("title", "No title")
        year = item.get("year", "No year")
        link = item.get("link", "No link")
        
        markdown_str += f"{i}. {year}: [{title}]({link})\n\n"
        i += 1


    return markdown_str


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
    
    # Query to get the top 5 results
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax=5&api_key={st.secrets['pubmed_api_key']}"
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

your_question = st.text_input("Your question for PubMed", placeholder="Enter your question here")
st.session_state.your_question = your_question

if st.button("Convert to PubMed Search Terms"):
    messages_search_terms = [{'role': 'system', 'content': search_term_system_prompt},
        {'role': 'user', 'content': st.session_state.your_question}]
    search_terms = create_chat_completion(messages_search_terms)                            
    st.write(f'Here are your search terms: \n\n {search_terms.choices[0].message.content}')      
                                    
    st.session_state.search_terms = search_terms
    
    articles, abstracts = pubmed_abstracts(search_terms.choices[0].message.content)
    
    articles = json_to_markdown(articles)
    st.write(articles)
    st.write(abstracts)