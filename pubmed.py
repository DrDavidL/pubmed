import requests
import streamlit as st
import time
import xml.etree.ElementTree as ET

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
    
    # Query to get the top results
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax={max_results}&api_key={st.secrets['pubmed_api_key']}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Check if no results were returned
        if 'count' in data['esearchresult'] and int(data['esearchresult']['count']) == 0:
            st.write("No results found. Try a different search or try again after re-loading the page.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching search results: {e}")
        return []

    ids = data['esearchresult']['idlist']
    if not ids:
        st.write("No results found.")
        return []

    # Fetch details and abstracts for all IDs
    id_str = ",".join(ids)
    details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id_str}&retmode=json&api_key={st.secrets['pubmed_api_key']}"
    abstracts_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_str}&retmode=xml&rettype=abstract&api_key={st.secrets['pubmed_api_key']}"

    articles = []

    try:
        # Fetch article details
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        details_data = details_response.json()
        
        # Fetch article abstracts
        abstracts_response = requests.get(abstracts_url)
        abstracts_response.raise_for_status()
        abstracts_data = abstracts_response.text

        for id in ids:
            if 'result' in details_data and str(id) in details_data['result']:
                article = details_data['result'][str(id)]
                year = article['pubdate'].split(" ")[0]
                if year.isdigit():
                    abstract = extract_abstract_from_xml(abstracts_data, id)
                    articles.append({
                        'title': article['title'],
                        'year': year,
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{id}",
                        'abstract': abstract.strip() if abstract.strip() else "No abstract available"
                    })
            else:
                st.warning(f"Details not available for ID {id}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching details or abstracts: {e}")

    return articles

def extract_abstract_from_xml(xml_data, pmid):
    # Parse the XML data to find the abstract text for the given PubMed ID (pmid)
    root = ET.fromstring(xml_data)
    for article in root.findall(".//PubmedArticle"):
        medline_citation = article.find("MedlineCitation")
        if medline_citation:
            pmid_element = medline_citation.find("PMID")
            if pmid_element is not None and pmid_element.text == pmid:
                abstract_elements = medline_citation.findall(".//AbstractText")
                abstract_text = ""
                for elem in abstract_elements:
                    abstract_text += ET.tostring(elem, encoding='unicode', method='text')
                return abstract_text
    return "No abstract available"

# Example usage in a Streamlit app
st.title("PubMed Article Search")

search_terms = st.text_input("Enter search terms:")
search_type = st.selectbox("Select search type:", ["all", "clinical trials", "reviews"])
max_results = st.number_input("Max results:", min_value=1, max_value=100, value=5)

if st.button("Search"):
    articles = pubmed_abstracts(search_terms, search_type, max_results)
    for article in articles:
        st.markdown(f"### [{article['title']}]({article['link']})")
        st.write(f"Year: {article['year']}")
        if article['abstract']:
            st.write(article['abstract'])
        else:
            st.write("No abstract available")
