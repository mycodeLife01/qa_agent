from loguru import logger

logger.add("logs/qa_agent.log", level="DEBUG")

def setup_logger():
    return logger