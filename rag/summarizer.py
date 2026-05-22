import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def textrank_summary(text: str, max_sentences: int = 3) -> str:
    """Generate summary using TextRank algorithm"""
    if not text or not text.strip():
        return "No content to summarize."

    # Tokenize into sentences
    sentences = sent_tokenize(text)

    if len(sentences) <= max_sentences:
        return text

    # Create sentence embeddings
    stop_words = set(stopwords.words('english'))
    sentence_embeddings = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words]
        sentence_embeddings.append(words)

    # Create similarity matrix
    n_sentences = len(sentences)
    similarity_matrix = np.zeros((n_sentences, n_sentences))

    for i in range(n_sentences):
        for j in range(n_sentences):
            if i != j:
                similarity = calculate_similarity(sentence_embeddings[i], sentence_embeddings[j])
                similarity_matrix[i][j] = similarity

    # Create graph and apply PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Select top sentences
    top_sentences = []
    for i in range(min(max_sentences, len(ranked_sentences))):
        top_sentences.append(ranked_sentences[i][1])

    # Return sentences in original order
    summary = " ".join(top_sentences)
    return summary


def calculate_similarity(sent1: list, sent2: list) -> float:
    """Calculate similarity between two sentences"""
    if not sent1 or not sent2:
        return 0.0

    all_words = list(set(sent1 + sent2))

    vector1 = [1 if word in sent1 else 0 for word in all_words]
    vector2 = [1 if word in sent2 else 0 for word in all_words]

    # Cosine similarity
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = sum(v ** 2 for v in vector1) ** 0.5
    magnitude2 = sum(v ** 2 for v in vector2) ** 0.5

    if magnitude1 * magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)