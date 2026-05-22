import re
from typing import List, Dict


def smart_answer(query: str, context: str, config: Dict = None) -> str:
    """Generate intelligent, concise answers using rule-based approach"""

    # First, limit the context to avoid processing huge chunks
    context = context[:2000]  # Only process first 2000 characters

    # Clean and prepare text
    query = query.lower().strip()
    sentences = split_into_sentences(context)

    # Find the best answer
    best_answer = find_best_answer(query, sentences)

    # Ensure answer is concise (max 150 characters)
    if len(best_answer) > 150:
        best_answer = best_answer[:147] + "..."

    return best_answer


def split_into_sentences(text: str) -> List[str]:
    """Split text into clean, short sentences"""
    # Split by periods, exclamation marks, question marks
    sentences = re.split(r'[.!?]+', text)

    # Clean and filter
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if 10 < len(sentence) < 300:  # Only keep reasonable length sentences
            clean_sentences.append(sentence)

    return clean_sentences[:10]  # Only consider first 10 sentences


def find_best_answer(query: str, sentences: List[str]) -> str:
    """Find the most relevant sentence and make it concise"""

    if not sentences:
        return "I couldn't find relevant information."

    query_words = set(query.lower().split())
    best_sentence = ""
    best_score = 0

    # Score each sentence
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        score = len(query_words.intersection(sentence_words))

        # Bonus for exact matches
        if query.lower() in sentence.lower():
            score += 5

        if score > best_score:
            best_score = score
            best_sentence = sentence

    # If no good match, use the first sentence
    if best_score == 0:
        best_sentence = sentences[0]

    # Make the answer concise and focused
    return create_concise_answer(query, best_sentence)


def create_concise_answer(query: str, sentence: str) -> str:
    """Create a concise answer from the best sentence"""

    # For "what is" questions, try to extract definition
    if query.startswith('what is') or 'about' in query:
        # Look for key phrases that define things
        if ' is ' in sentence:
            parts = sentence.split(' is ')
            if len(parts) > 1:
                definition = parts[1].split('.')[0]  # Get first part after "is"
                if len(definition) > 10:
                    return definition.strip()

        # Look for descriptions
        if 'enables' in sentence or 'provides' in sentence or 'allows' in sentence:
            # Extract the action part
            for keyword in ['enables', 'provides', 'allows']:
                if keyword in sentence:
                    idx = sentence.find(keyword)
                    excerpt = sentence[max(0, idx - 20):idx + 60]
                    return excerpt.strip()

    # For technology questions
    if 'technolog' in query or 'tools' in query or 'services' in query:
        # Extract technology mentions
        tech_terms = []
        words = sentence.split()
        for i, word in enumerate(words):
            if any(tech in word.lower() for tech in ['aws', 'google', 'azure', 'api', 'cloud']):
                # Get context around tech terms
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                tech_phrase = ' '.join(words[start:end])
                tech_terms.append(tech_phrase)

        if tech_terms:
            return f"Technologies include: {', '.join(tech_terms[:3])}"

    # Default: return first 100 characters of the sentence
    if len(sentence) > 100:
        return sentence[:97] + "..."
    else:
        return sentence