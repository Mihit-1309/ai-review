import os
from components.csv_loader import specialized_csv_load,create_text_chunks
from components.vector_store import save_vector_store
from dotenv import load_dotenv
load_dotenv()

from common.logger import get_logger
from common.custom_exception import CustomException

logger = get_logger(__name__)

column_to_index = ["product_name","review_title","review_text"]
def process_and_store_pdfs():
    try:
        logger.info("Making the vector store.....")

        documents = specialized_csv_load("data/datas",content_columns=column_to_index)

        text_chunks = create_text_chunks(documents)

        save_vector_store(text_chunks)

        logger.info("Vector store created successfullly")

    except Exception as e:
        error_message = CustomException("Failed to process and store PDFs")
        logger.error(str(error_message))


if __name__ == "__main__":
    process_and_store_pdfs()