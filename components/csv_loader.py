import os
import warnings
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.custom_exception import CustomException
from common.logger import get_logger
from config.config import CHUNK_OVERLAP, CHUNK_SIZE
from components.database import reviews_collection
import uuid

logger = get_logger(__name__)
def load_csv_to_db(df):
    records = []

    for _, row in df.iterrows():
        records.append({
            "review_id": str(uuid.uuid4()),
            "wsid": row["WSID"],                 # 🔴 CAPS FIX
            "product_id": str(row["product_id"]),
            "product_name": row["product_name"],
            "review_title": row["review_title"],
            "review_text": row["review_text"],
            "rating": int(row["rating"]),
            "embedded": False
        })

    if records:
        reviews_collection.insert_many(records)


        
def basic_csv_load(file_path):
    """
    Loads a CSV where every column is included in the content with error handling.
    """
    try:
        path_obj = Path(file_path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"The file at {path_obj} does not exist.")
            
        logger.info(f"Attempting to load CSV from: {path_obj}")
        # Added encoding='utf-8-sig' to handle files saved from Excel with BOM
        loader = CSVLoader(file_path=str(path_obj), encoding='utf-8-sig')
        data = loader.load()
        
        logger.info(f"Successfully loaded {len(data)} documents.")
        return data

    except Exception as e:
        logger.error(f"An unexpected error occurred while loading CSV: {e}")
        return []

def specialized_csv_load(file_path, content_columns, metadata_columns=None):
    """
    Loads a CSV and handles specific columns for content and metadata.
    Args:
        file_path: Path to your .csv file or a directory containing .csv files.
        content_columns: A list of column names to include in the page_content.
        metadata_columns: A list of column names to keep as metadata.
    """
    try:
        all_docs = []
        paths_to_process = []
        
        # Using .resolve() to get absolute path and handle cross-platform separators
        base_path = Path(file_path).resolve()

        if not base_path.exists():
            # Fallback check: if 'data/datas' was meant to be 'data/datas.csv'
            if not base_path.suffix and base_path.with_suffix('.csv').exists():
                base_path = base_path.with_suffix('.csv')
                logger.info(f"Path did not exist, but found matching .csv: {base_path}")
            else:
                logger.error(f"Path does not exist: {file_path} (Resolved: {base_path})")
                return []

        # Handle Directory vs File logic
        if base_path.is_dir():
            logger.info(f"Path is a directory. Searching for CSVs in: {base_path}")
            paths_to_process = list(base_path.glob("*.csv"))
            if not paths_to_process:
                # Fallback: check all files in directory if no .csv found
                paths_to_process = [f for f in base_path.iterdir() if f.is_file()]
        else:
            paths_to_process = [base_path]

        if not paths_to_process:
            logger.warning(f"No files found to process at path: {base_path}")
            return []

    except Exception as e:
        logger.error(f"Error while scanning directory/file: {e}")
        return []

    for p in paths_to_process:
        str_path = str(p.absolute())
        try:
            logger.info(f"Processing file: {str_path}")
            
            # Encapsulating the load in different encodings to handle Excel/Windows CSVs
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            data = None
            last_error = ""

            for enc in encodings:
                try:
                    loader = CSVLoader(
                        file_path=str_path,
                        encoding=enc,
                        csv_args={
                            'delimiter': ',',
                            'quotechar': '"'
                        }
                    )
                    data = loader.load()
                    if data:
                        logger.info(f"Successfully loaded file using {enc} encoding.")
                        break
                except Exception as encoding_err:
                    last_error = str(encoding_err)
                    continue
            
            if data is None:
                raise Exception(f"Failed to parse CSV with standard encodings. Last error: {last_error}")

            # Manual filtering logic to strictly control page_content
            if content_columns and data:
                # for doc in data:
                #     lines = doc.page_content.split('\n')
                #     filtered_lines = []
                #     for line in lines:
                #         if ':' in line:
                #             # Split on first colon to get header
                #             header = line.split(':', 1)[0].strip()
                #             if header in content_columns:
                #                 filtered_lines.append(line)
                    
                #     doc.page_content = "\n".join(filtered_lines)
            
                for doc in data:
                    lines = doc.page_content.split('\n')

                    content_lines = []
                    metadata = {}

                    for line in lines:
                        if ':' not in line:
                            continue

                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # Content columns → page_content
                        if key in content_columns:
                            content_lines.append(f"{key}: {value}")

                        # Metadata columns → metadata dict
                        if key == "WSID":
                            metadata["WSID"] = value

                        elif key == "product_id":
                            try:
                                metadata["product_id"] = int(value)
                            except:
                                metadata["product_id"] = value

                        elif key == "rating":
                            try:
                                metadata["rating"] = int(value)
                            except:
                                metadata["rating"] = value

                        elif key == "product_name":
                            metadata["product_name"] = value

                        elif key in ["Store name", "store_name"]:
                            metadata["store_name"] = value

                        elif key == "review_date":
                            metadata["review_date"] = value

                    # Final assignment
                    doc.page_content = "\n".join(content_lines)
                    doc.metadata = metadata
            logger.debug(f"Metadata assigned: {doc.metadata}")

            all_docs.extend(data)
            logger.info(f"Successfully processed {len(data)} rows from {p.name}")

        except Exception as e:
            logger.error(f"Failed to load file {str_path}: {str(e)}")
            continue

    if not all_docs:
        logger.warning(f"Specialized CSV load resulted in 0 documents from {len(paths_to_process)} files.")
    else:
        logger.info(f"Specialized CSV load completed. Total documents: {len(all_docs)}")
        
    return all_docs

def create_text_chunks(documents):
    """
    Splits documents into smaller chunks for vector storage.
    """
    if not documents:
        logger.warning("No Documents provided to create_text_chunks.")
        return []
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Generated {len(text_chunks)} text chunks.")
        return text_chunks

    except Exception as e:
        error_msg = f"Failed to create text chunks. Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise CustomException(error_msg)