import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from langchain_text_splitters import HTMLHeaderTextSplitter

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NPS_API_KEY = os.getenv("NPS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
ARTICLE_LIMIT = os.getenv("ARTICLE_LIMIT") # Limit the number of articles per park
PARK_LIMIT = os.getenv("PARK_LIMIT") # Limit the number of parks

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV
        )
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Set up our Pinecone vector store
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index, embedding_model, "text")

# Define metadata fields for the documents we will store in Pinecone
metadata_field_info = [
    AttributeInfo(
        name="park_name",
        description="The name of the national park",
        type="string",
    ),
    AttributeInfo(
        name="article_title",
        description="The title of the article",
        type="string",
    ),
    AttributeInfo(
        name="url",
        description="The URL of the article",
        type="string",
    ),
]

document_content_description = "Text of the NPS article"

# Set up the language model
llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)

# Set up the retriever
retriever = SelfQueryRetriever.from_llm(
    llm, vector_store, document_content_description, metadata_field_info, verbose=True
)

def get_nps_data(api_key):
    """Fetches data from the National Park Service (NPS) API and stores it in Pinecone"""
    base_url = 'https://developer.nps.gov/api/v1/'
    headers = {'X-Api-Key': api_key}

    def get_parks():
        """Fetches a list of parks from the NPS API."""
        endpoint = 'parks'
        params = {'limit': PARK_LIMIT, 'fields': 'images'}
        response = requests.get(base_url + endpoint, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            print(f"Error fetching parks: {response.status_code} - {response.json().get('message')}")
            return []

    def get_articles(park_code):
        """Fetches articles related to a park from the NPS API."""
        endpoint = 'articles'
        params = {'parkCode': park_code, 'limit': ARTICLE_LIMIT}
        response = requests.get(base_url + endpoint, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            print(f"Error fetching articles for {park_code}: {response.status_code} - {response.json().get('message')}")
            return []

    def fetch_article_text(url):
        """Fetches the text content of an article from the NPS website."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                main_content_div = soup.find('div', class_='MainContent') #Pluck out the main content div
                if main_content_div:
                    html_content = main_content_div.prettify()
                    return html_content
                else:
                    print("MainContent div not found.")
                    return ""
            else:
                print(f"Error fetching article text: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Exception occurred while fetching article text: {e}")
            return ""

    # Set up the HTML header text splitter
    # We will split the HTML content based on the headers in the article to overcome storage limitations in Pinecone
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    parks = get_parks()
    all_parks_articles = {}

    article_count = 0

    # For each park, get related articles and their text content
    # Insert the text content into Pinecone
    for park in parks:
        park_code = park['parkCode']
        park_name = park['fullName']
        articles = get_articles(park_code)
        articles_with_text = []
        for article in articles:
            html_content = fetch_article_text(article['url'])
            if html_content:
                split_documents = html_splitter.split_text(html_content)
                split_texts = [doc.page_content for doc in split_documents]
                article['text'] = " ".join(split_texts)
                articles_with_text.append(article)
                # Insert article into Pinecone
                vector_store.add_texts(split_texts, [{"park_name": park_name, "article_title": article['title'], "url": article['url']}])
                article_count += 1
                print(f"Inserted article: {article['title']} for park: {park_name}")
        all_parks_articles[park_name] = articles_with_text

    print(f"Inserted a total of {article_count} articles into Pinecone.")
    return all_parks_articles

# Fetch data from NPS API and store in Pinecone
parks_articles = get_nps_data(NPS_API_KEY)

# Now you can use the retriever to query the articles
query = "Find articles about Alcatraz Island."
results = retriever.invoke(query)

for result in results:
    print(f"Title: {result.metadata['article_title']}")
    print(f"Park: {result.metadata['park_name']}")
    print(f"URL: {result.metadata['url']}")
    print(f"Text: {result.page_content[:200]}...")
    print("\n")
