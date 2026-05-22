try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    import torch

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def huggingface_answer(query: str, context: str, config: Dict = None) -> str:
    """Generate answer using free Hugging Face models"""

    if not HF_AVAILABLE:
        return fallback_textrank(context)

    try:
        # Use a free question-answering model
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )

        # Truncate context if too long (BERT has token limits)
        max_context = 500
        if len(context) > max_context:
            context = context[:max_context]

        result = qa_pipeline(question=query, context=context)

        answer = result['answer']
        confidence = result['score']

        # If confidence is low, provide more context
        if confidence < 0.3:
            return f"Based on the available information: {answer}. {context[:200]}..."
        else:
            return answer

    except Exception as e:
        print(f"Hugging Face error: {e}")
        return fallback_textrank(context)


def fallback_textrank(context: str) -> str:
    """Fallback to simple summary"""
    sentences = context.split('.')
    important_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
    return '. '.join(important_sentences) + '.'