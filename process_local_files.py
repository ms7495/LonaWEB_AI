import argparse
import logging
import sys
from pathlib import Path

# Add the main directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
main_dir = current_dir / "main"
if str(main_dir) not in sys.path:
    sys.path.insert(0, str(main_dir))

# Import your DocuChat engine
from core.rag_engine import DocuChatEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalFileProcessor:
    """Process local files and add them to the vector database"""

    def __init__(self):
        self.engine = DocuChatEngine()
        self.supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv'}

    def process_single_file(self, file_path: Path) -> bool:
        """Process a single file"""
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False

            if file_path.suffix.lower() not in self.supported_extensions:
                logger.warning(f"Skipping unsupported file: {file_path}")
                return False

            logger.info(f"Processing file: {file_path}")

            # Create a mock uploaded file object
            class MockUploadedFile:
                def __init__(self, file_path: Path):
                    self.name = file_path.name
                    self.size = file_path.stat().st_size
                    self._content = file_path.read_bytes()

                def getvalue(self):
                    return self._content

            mock_file = MockUploadedFile(file_path)

            # Process using the engine
            result = self.engine.process_uploaded_file(mock_file)

            if result["success"]:
                logger.info(f"✅ Successfully processed: {file_path.name}")
                logger.info(f"   Created {result['chunks_created']} chunks")
                return True
            else:
                logger.error(f"❌ Failed to process {file_path.name}: {result['error']}")
                return False

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False

    def process_directory(self, directory_path: Path, recursive: bool = True) -> dict:
        """Process all supported files in a directory"""
        results = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "files": []
        }

        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results

        logger.info(f"Processing directory: {directory_path}")

        # Find all files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                if file_path.suffix.lower() in self.supported_extensions:
                    if self.process_single_file(file_path):
                        results["processed"] += 1
                        results["files"].append(str(file_path))
                    else:
                        results["failed"] += 1
                else:
                    results["skipped"] += 1
                    logger.debug(f"Skipped unsupported file: {file_path}")

        return results

    def get_current_documents(self):
        """Get information about currently stored documents"""
        try:
            stats = self.engine.get_document_stats()
            logger.info("Current documents in database:")
            logger.info(f"  Total documents: {stats['total_documents']}")
            logger.info(f"  Total chunks: {stats['total_chunks']}")

            for doc in stats['documents']:
                logger.info(f"  - {doc['name']}: {doc['chunks']} chunks, {doc['pages']} pages")

            return stats
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return None

    def clear_database(self):
        """Clear all documents from the database"""
        try:
            result = self.engine.clear_documents()
            if result["success"]:
                logger.info("✅ Database cleared successfully")
                return True
            else:
                logger.error(f"❌ Failed to clear database: {result['error']}")
                return False
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process local files for DocuChat")
    parser.add_argument("path", nargs="?", help="Path to file or directory to process")
    parser.add_argument("--folder", "-f", help="Folder to process")
    parser.add_argument("--recursive", "-r", action="store_true", default=True,
                        help="Process subdirectories recursively (default: True)")
    parser.add_argument("--clear", "-c", action="store_true",
                        help="Clear database before processing")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Show current database status")
    parser.add_argument("--examples", action="store_true",
                        help="Show usage examples")

    args = parser.parse_args()

    if args.examples:
        print_examples()
        return

    # Initialize processor
    processor = LocalFileProcessor()

    # Show status if requested
    if args.status:
        processor.get_current_documents()
        return

    # Clear database if requested
    if args.clear:
        if input("Are you sure you want to clear the database? (y/N): ").lower() == 'y':
            processor.clear_database()
        else:
            logger.info("Database clear cancelled")
            return

    # Determine path to process
    target_path = None
    if args.path:
        target_path = Path(args.path)
    elif args.folder:
        target_path = Path(args.folder)
    else:
        # Default to local_docs folder
        target_path = Path("local_docs")
        if not target_path.exists():
            logger.info("Creating local_docs folder...")
            target_path.mkdir(exist_ok=True)
            logger.info("📁 Created 'local_docs' folder. Add your files there and run again.")
            return

    if not target_path.exists():
        logger.error(f"Path not found: {target_path}")
        return

    # Process files
    logger.info("🚀 Starting file processing...")

    if target_path.is_file():
        # Process single file
        success = processor.process_single_file(target_path)
        if success:
            logger.info("✅ File processed successfully!")
        else:
            logger.error("❌ File processing failed!")
    else:
        # Process directory
        results = processor.process_directory(target_path, args.recursive)

        logger.info("📊 Processing Summary:")
        logger.info(f"  ✅ Processed: {results['processed']} files")
        logger.info(f"  ❌ Failed: {results['failed']} files")
        logger.info(f"  ⏭️ Skipped: {results['skipped']} files")

        if results['processed'] > 0:
            logger.info("🎉 Files successfully added to the vector database!")
            logger.info("You can now query these documents using the DocuChat interface.")

    # Show final status
    logger.info("\n" + "=" * 50)
    processor.get_current_documents()


def print_examples():
    """Print usage examples"""
    examples = """
📖 DocuChat Local File Processing Examples

1. Process a single file:
   python process_local_files.py /path/to/document.pdf
   python process_local_files.py ./my_document.docx

2. Process all files in a folder:
   python process_local_files.py /path/to/documents/
   python process_local_files.py --folder ./my_docs/

3. Process with options:
   python process_local_files.py ./docs/ --clear     # Clear database first
   python process_local_files.py --status            # Show current status

4. Recommended workflow:
   mkdir local_docs
   # Copy your files to local_docs/
   python process_local_files.py local_docs

5. File structure example:
   local_docs/
   ├── research_paper.pdf
   ├── company_docs/
   │   ├── handbook.docx
   │   └── policies.pdf
   └── data.xlsx

Supported formats: PDF, DOCX, DOC, TXT, XLSX, XLS, CSV
"""
    print(examples)


if __name__ == "__main__":
    main()
