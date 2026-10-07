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


def textrank_summary(text: str, max_sentences: int = 3, query: str = None) -> str:
    """Generate summary using TextRank.

    If a query is given, PageRank is personalized toward sentences that share
    words with it, so different questions get different answers.
    """
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

    personalization = None
    if query:
        query_words = {w for w in word_tokenize(query.lower()) if w.isalnum() and w not in stop_words}
        if query_words:
            overlap = [len(query_words & set(words)) for words in sentence_embeddings]
            if sum(overlap) > 0:
                # Small floor so non-matching sentences can still be reached
                personalization = {i: o + 0.05 for i, o in enumerate(overlap)}

    try:
        scores = nx.pagerank(nx_graph, personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        scores = {i: 1.0 / n_sentences for i in range(n_sentences)}

    # Boost sentences that actually mention the question's words
    if personalization:
        max_overlap = max(overlap)
        scores = {i: scores[i] * (1 + 3 * overlap[i] / max_overlap) for i in scores}

    # Pick the top sentences, then put them back in original order
    top_idx = sorted(range(n_sentences), key=lambda i: scores[i], reverse=True)[:max_sentences]
    summary = " ".join(sentences[i] for i in sorted(top_idx))
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