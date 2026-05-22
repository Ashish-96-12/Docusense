
import os
import importlib
import sys


def check_updates():
    errors = []

    # Check file existence
    required_files = [
        'app.py',
        'config.yaml',
        'rag/pipeline.py',
        'rag/index.py',
        'rag/chunkers.py',
        'rag/summarizer.py',
        'rag/utils.py',
        'static/docusense_web.html'
    ]

    print("Checking file existence...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} MISSING")
            errors.append(f"Missing file: {file}")

    # Check key functions
    print("\nChecking implementation...")

    try:
        # Check if pipeline has correct signature
        from rag.pipeline import RagPipeline
        import inspect

        # Check answer method signature
        answer_sig = str(inspect.signature(RagPipeline.answer))
        if 'doc_id' in answer_sig:
            print("✅ pipeline.py has doc_id parameter")
        else:
            print("❌ pipeline.py missing doc_id parameter")
            errors.append("Pipeline answer method missing doc_id parameter")

    except Exception as e:
        print(f"❌ Error checking pipeline: {e}")
        errors.append(f"Pipeline import error: {e}")

    try:
        # Check if index.py has IndexManager
        from rag.index import IndexManager
        if hasattr(IndexManager, '_search_document'):
            print("✅ index.py has document-specific search")
        else:
            print("❌ index.py missing _search_document method")
            errors.append("IndexManager missing document-specific methods")
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        errors.append(f"Index import error: {e}")

    # Check storage directories
    print("\nChecking directories...")
    dirs = ['storage', 'storage/corpus', 'storage/chunks', 'storage/indices', 'static']
    for dir in dirs:
        if os.path.exists(dir):
            print(f"✅ {dir} exists")
        else:
            print(f"⚠️  {dir} missing (will be created on first use)")

    # Summary
    print("\n" + "=" * 50)
    if not errors:
        print("✅ ALL CHECKS PASSED! Your system is properly updated.")
    else:
        print("❌ Issues found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease update the files mentioned above.")

    return len(errors) == 0


if __name__ == "__main__":
    check_updates()