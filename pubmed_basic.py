import streamlit as st
from openai import OpenAI
from prompts import search_term_system_prompt, disclaimer
import requests
import time
import json
import markdown2
from bs4 import BeautifulSoup
from fpdf import FPDF

st.set_page_config(page_title='PubMed Researcher', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("PubMed Researcher")
st.write("ALPHA version 0.5")


with st.expander('About PubMed Researcher - Important Disclaimer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.info(disclaimer)
    st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.5, 0.01)
    st.write("Last updated 6/20/24")

class PDF(FPDF):
    def __init__(self, title='Document'):
        super().__init__()
        self.title = title
    
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, self.title, 0, 1, 'C')

    def chapter_title(self, txt, level):
        if level == 1:
            self.set_font('Arial', 'B', 16)
        elif level == 2:
            self.set_font('Arial', 'B', 14)
        elif level == 3:
            self.set_font('Arial', 'B', 12)
        self.cell(0, 10, txt.encode('latin-1', 'replace').decode('latin-1'), 0, 1)

    def chapter_body(self, txt):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, txt.encode('latin-1', 'replace').decode('latin-1'))

    def add_list(self, items, is_ordered):
        self.set_font('Arial', '', 12)
        for i, item in enumerate(items):
            if is_ordered:
                line = f"{i+1}. {item}"
            else:
                line = f"• {item}"
            self.cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'), 0, 1)

def html_to_pdf(html_content, name):
    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract title for the document
    case_title_tag = soup.find("h1")
    case_title = case_title_tag.get_text() if case_title_tag else "Document"
    
    # Create PDF instance with dynamic title
    pdf = PDF(title=case_title)
    pdf.add_page()

    # Process each section of the HTML
    for element in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "li", "hr"]):
        if element.name == "h1":
            pdf.chapter_title(element.get_text(), level=1)
        elif element.name == "h2":
            if "Patient Door Chart" in element.get_text():
                pdf.add_page()
            pdf.chapter_title(element.get_text(), level=2)

            # Add content within h2 tag
            for child in element.find_all_next():
                if child.name == "h2":  # Stop when next h2 is found
                    break
                if child.name == "p":
                    pdf.chapter_body(child.get_text())
                elif child.name == "ul":
                    items = [li.get_text() for li in child.find_all("li")]
                    pdf.add_list(items, is_ordered=False)
                elif child.name == "ol":
                    items = [li.get_text() for li in child.find_all("li")]
                    pdf.add_list(items, is_ordered=True)
                elif child.name == "h3":
                    pdf.chapter_title(child.get_text(), level=3)
        elif element.name == "h3":
            pdf.chapter_title(element.get_text(), level=3)
        elif element.name == "p":
            pdf.chapter_body(element.get_text())
        elif element.name == "ul":
            items = [li.get_text() for li in element.find_all("li")]
            pdf.add_list(items, is_ordered=False)
        elif element.name == "ol":
            items = [li.get_text() for li in element.find_all("li")]
            pdf.add_list(items, is_ordered=True)
        elif element.name == "hr":
            pdf.add_page()
    
    # Output the PDF
    pdf.output(name)

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
        link = item.get("link", "No link")
        year = item.get("year", "No year")

        
        markdown_str += f"{i}. {year}: [{title}]({link})\n\n"
        i += 1


    return markdown_str


# Longer approach to handle cases with no results
def pubmed_abstracts(search_terms, search_type="all", max_results=5):
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
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax={max_results}&api_key={st.secrets['pubmed_api_key']}"
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
            st.error(f"No abstract avaiable for {id}: {e}")
            time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    return articles, abstracts

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

if "articles" not in st.session_state:
    st.session_state.articles = []
if "articles_markdown" not in st.session_state:
    st.session_state.articles_markdown = ""
if "abstracts" not in st.session_state:
    st.session_state.abstracts = []

original_question = st.text_input("Your question for PubMed", placeholder="Enter your question here")
search_type = st.radio("Select an Option", ("all", "clinical trials", "reviews"), horizontal=True)
number_of_results = st.slider("Number of Results", 1, 20, 5, 1)

if st.button("Convert to PubMed Search Terms"):
    messages_search_terms = [{'role': 'system', 'content': search_term_system_prompt},
        {'role': 'user', 'content': original_question}]
    search_terms = create_chat_completion(messages_search_terms)      
    with st.popover("Search performed"):                      
        st.write(f'Here are your search terms: \n\n {search_terms.choices[0].message.content}')     
        st.write(f'Type of search: {search_type}')
        st.write(f'Number of results: {number_of_results}') 
                                    
    st.session_state.search_terms = search_terms.choices[0].message.content
    
    articles, abstracts = pubmed_abstracts(search_terms.choices[0].message.content, search_type=search_type, max_results=number_of_results)
    st.session_state.articles = articles
    st.session_state.abstracts = abstracts
    
    # 
    # st.write(f'Here are the articles: \n\n {articles_markdown}')
    # st.write(f'Here are the abstracts: \n\n {"\n".join(abstracts)}')


    articles_markdown = ""
    for article, abstract in zip(st.session_state.articles, st.session_state.abstracts):
        with st.expander(f"**{article['title']}** ({article['year']})"):
            st.write(f"{article['title']} ({article['year']})")
            st.write(abstract)
            st.write(f"Link: [{article['link']}]({article['link']})")
            st.write("----")
            articles_markdown += f"{article['title']} ({article['year']})\n\n{abstract}\n\nLink: [{article['link']}]({article['link']})\n\n----\n\n"
    
    st.session_state.articles_markdown = articles_markdown
if st.session_state.articles_markdown != "":

    articles_html = markdown2.markdown(st.session_state.articles_markdown, extras=["tables"])
        # st.download_button('Download HTML Case file', html, f'case.html', 'text/html')
        
    # st.info("Download the Current Case:")
    if st.checkbox("Download Article List PDF file"):
        html_to_pdf(f"<h1>PubMed Search</h1> <h2>Search: {st.session_state.search_terms} </h2> \n\n {articles_html}", 'article_list.pdf')
        with open("article_list.pdf", "rb") as f:
            st.download_button("Article List PDF", f, "article_list.pdf")