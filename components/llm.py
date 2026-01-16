from config.config import GROQ_API_KEY, GROQ_MODEL_NAME
from common.logger import get_logger
from common.custom_exception import CustomException
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
load_dotenv()
logger = get_logger(__name__)

def load_llm(model_name: str = GROQ_MODEL_NAME,groq_api_key: str = GROQ_API_KEY):
    try:
        logger.info("Loading LLM from GROQ using LLama3 model...")

        llm = ChatGroq(
            groq_api_key = groq_api_key,
            model_name = model_name,
            temperature=0,
            max_tokens=2048
        )

        logger.info("LLM loaded successfully from GROQ.")
        return llm
    
    except Exception as e:
        error_message = CustomException("Failed to load an LLM from GROQ")
        logger.error(str(error_message))
        return None