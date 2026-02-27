import requests
from bs4 import BeautifulSoup
from common.logger import get_logger

logger = get_logger(__name__)

# Cache per product URL
WEBSITE_CACHE = {}


def get_website_content(product_url: str):
    """
    Fetch and cache product website content dynamically.
    """

    if product_url in WEBSITE_CACHE:
        logger.info("Website cache hit | url=%s", product_url)
        return WEBSITE_CACHE[product_url]

    logger.info("Fetching website content | url=%s", product_url)

    try:
        response = requests.get(product_url, timeout=8)

        if response.status_code != 200:
            logger.warning(
                "Website request failed | url=%s | status=%s",
                product_url,
                response.status_code
            )
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        clean_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )

        WEBSITE_CACHE[product_url] = clean_text

        logger.info(
            "Website content cached successfully | url=%s | chars=%d",
            product_url,
            len(clean_text)
        )

        return clean_text

    except Exception as e:
        logger.error(
            "Website scraping failed | url=%s | error=%s",
            product_url,
            str(e),
            exc_info=True
        )
        return ""
