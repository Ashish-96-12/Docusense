#!/usr/bin/env python3
"""
Test script for DocuSense - Run this to verify everything works
"""

import os
from config import CONFIG
from rag.pipeline import RagPipeline


def create_test_document():
    """Create a test document"""
    test_content = """
    DocuSense is an intelligent document analysis system that uses RAG (Retrieval-Augmented Generation) 
    to process and analyze documents. The system can handle PDF, DOCX, and TXT files.

    Key Features:
    - Secure offline processing ensures data privacy
    - BM25 retrieval for finding relevant information  
    - TextRank summarization for generating concise answers
    - Modular architecture for easy extension

    The system works by first ingesting documents and breaking them into chunks. These chunks are then 
    indexed using BM25 for efficient retrieval. When a user asks a question, the system finds the most 
    relevant chunks and uses TextRank to generate a summary answer.

    DocuSense is designed for enterprise use where data security and accurate information retrieval 
    are critical requirements. The offline processing ensures that sensitive documents never leave 
    the local environment.
    """

    # Save test document
    os.makedirs("storage/corpus", exist_ok=True)
    test_file = "storage/corpus/docusense_info.txt"

    with open(test_file, "w") as f:
        f.write(test_content)

    return test_file


def test_docusense():
    """Test the complete DocuSense pipeline"""
    print("🚀 Starting DocuSense Test...")
    print("=" * 50)

    try:
        # Create test document
        print("1. Creating test document...")
        test_file = create_test_document()
        print(f"   ✅ Created: {test_file}")

        # Initialize pipeline
        print("\n2. Initializing RAG pipeline...")
        pipeline = RagPipeline(CONFIG)
        print("   ✅ Pipeline initialized")

        # Ingest document
        print("\n3. Ingesting document...")
        doc_id = pipeline.ingest(test_file)
        print(f"   ✅ Document ingested with ID: {doc_id}")

        # Test queries
        test_queries = [
            "What is DocuSense?",
            "What are the key features?",
            "How does the retrieval system work?",
            "Why is offline processing important?"
        ]

        print("\n4. Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            answer, evidence = pipeline.answer(query)
            print(f"   Answer: {answer}")
            print(f"   Sources: {len(evidence)} chunks found")

        # Get stats
        print("\n5. System Statistics:")
        stats = pipeline.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n" + "=" * 50)
        print("🎉 All tests passed! DocuSense is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the web server")
        print("2. Visit http://localhost:8000 to see the API")
        print("3. Use POST /ingest to upload documents")
        print("4. Use POST /query to ask questions")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\nCheck that all dependencies are installed:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    test_docusense()