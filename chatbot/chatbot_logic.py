import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')
from pykrige.ok import OrdinaryKriging
import random
from django.conf import settings
import os
# Uncomment these lines if you haven't downloaded NLTK data before
# nltk.download("punkt")
# nltk.download("stopwords")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

# Preprocess text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stopwords]

# Extract keywords and index
def extract_keywords(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    keywords = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    return keywords

# Process user query and extract keywords
def process_query(query):
    preprocessed_query = preprocess_text(query)
    query_keywords = extract_keywords(" ".join(preprocessed_query))
    print("Query keywords: ", query_keywords)
    return query_keywords

# Segment the document into smaller parts
def segment_document(text):
    # Here, we split the document into sentences for finer granularity
    sentences = nltk.sent_tokenize(text)
    return sentences

# Search and retrieve relevant text
def search_relevant_text(query_keywords, document_text):
    segments = segment_document(document_text)
    vectorizer = TfidfVectorizer()
    doc_segments_matrix = vectorizer.fit_transform(segments)
    query_vector = vectorizer.transform([" ".join(query_keywords.keys())])
    similarity_scores = cosine_similarity(query_vector, doc_segments_matrix)
    
    most_similar_segment_idx = similarity_scores.argmax()
    most_similar_segment = segments[most_similar_segment_idx]
    return most_similar_segment

# Generate and send response
def generate_kriging_map(geojson_data, mineral, toposheet):
    features = [feature for feature in geojson_data['features'] if feature['properties']['toposheet'].lower() == toposheet.lower()]

    if not features:
        return "No data found for the specified toposheet."

    lons = [feature['geometry']['coordinates'][0] for feature in features]
    lats = [feature['geometry']['coordinates'][1] for feature in features]
    values = [feature['properties'][mineral] for feature in features if feature['properties'][mineral] is not None]

    if not values:
        return f"No data available for the mineral '{mineral}' in the specified toposheet."
    # else:
    #     return values

    # Downsample the data
    sample_size = min(30, len(lons))  # Use a maximum of 50 points for kriging
    sample_indices = random.sample(range(len(lons)), sample_size)
    lons = [lons[i] for i in sample_indices]
    lats = [lats[i] for i in sample_indices]
    values = [values[i] for i in sample_indices]

    grid_lon = np.linspace(min(lons), max(lons), 20)  # Reduced number of grid points
    grid_lat = np.linspace(min(lats), max(lats), 20)  # Reduced number of grid points
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    kriging = OrdinaryKriging(lons, lats, values, variogram_model='linear')
    z, ss = kriging.execute('grid', grid_lon[0], grid_lat[:, 0])
    z = z.reshape(grid_lon.shape)
    plt.figure(figsize=(6,4))
    plt.contourf(grid_lon, grid_lat, z, cmap='viridis')
    plt.colorbar(label=f'{mineral} Concentration')
    plt.scatter(lons, lats, c='red', marker='o', label='Sample Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Kriging Map for {mineral} in Toposheet {toposheet}')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def load_and_preprocess_geojson(geojson_path):
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    # Remove null values
    for feature in geojson_data['features']:
        feature['properties'] = {k: v for k, v in feature['properties'].items() if v is not None}
    
    return geojson_data


# Load and prepare document once (to avoid loading it on every request)
pdf_path = os.path.join(settings.BASE_DIR,'chatbot','mines.pdf')
document_text = extract_text_from_pdf(pdf_path)
preprocessed_document_text = preprocess_text(document_text)
document_keywords = extract_keywords(" ".join(preprocessed_document_text))
# geojson data
geojson_path = os.path.join(settings.BASE_DIR,'chatbot','55K03.geojson')
geojson_data = load_and_preprocess_geojson(geojson_path)




def generate_response(relevant_text):
    print('The path is:',os.path.join(settings.BASE_DIR,'chatbot','55K03.geojson'))
    return relevant_text
def get_response_for_query(query):
    print('The path is:',os.path.join(settings.BASE_DIR,'chatbot','55K03.geojson'))

    query_keywords = process_query(query)
    
    if 'kriging' in query.lower():
        # Extract mineral and toposheet from the query
        # mineral = None
        toposheet = None
        for word in query_keywords.keys():
            if word in ["sio2", "al2o3", "fe2o3", "tio2", "cao", "mgo", "mno", "na2o", "k2o", "p2o5", "loi", "ba", "ga", "sc", "v", "th", "pb", "ni", "co", "rb", "sr", "y", "zr", "nb", "cr", "cu", "zn", "au", "cs", "as_", "sb", "bi", "se", "ag", "be", "ge", "mo", "sn", "la", "ce", "pr", "nd", "sm", "eu", "tb", "gd", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w", "u", "pt", "pd" , "in_" , "f" ,"te" , "ti","hg","cd"]:
                mineral = word
            # else:
            #     mineral = None
            if word.startswith("55k"):  # Assuming toposheets are in this format
                toposheet = word
            
            
        
        if mineral and toposheet:
            return generate_kriging_map(geojson_data, mineral, toposheet)
        # elif mineral == None:
        #     return "This mineral is not found in toposheet 55K03"
        else:
            return "Please specify both a mineral and a toposheet number for kriging map generation."
    
    relevant_text = search_relevant_text(query_keywords, document_text)
    response = generate_response(relevant_text)
    return response


data = get_response_for_query("MMDR Act")
# typ = type(get_response_for_query("What is mines and minerals?"))
# print(typ, " and data is: ", data)
