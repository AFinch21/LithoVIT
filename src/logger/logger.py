import logging

# Define ANSI color codes for terminal text
RESET = "\x1b[0m"
COLORS = {
    'DEBUG': "\x1b[36m",      # Cyan
    'INFO': "\x1b[32m",       # Green
    'LLM': "\x1b[34m",    # Blue
    'WARNING': "\x1b[33m",    # Yellow
    'ERROR': "\x1b[31m",      # Red
    'CRITICAL': "\x1b[41m",   # Red Background
}

# Custom log level
LLM_LOG_LEVEL = 10
logging.addLevelName(LLM_LOG_LEVEL, "LLM")

# Custom formatter that adds color codes to log records
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Default log message format
        base_format = "%(asctime)s - %(levelname)s - %(message)s"
        
        # Check if there's a color associated with the log level
        if record.levelname in COLORS:
            # Format string with color codes applied to asctime and levelname
            colored_format = (
                COLORS[record.levelname] + "%(asctime)s - %(levelname)s" + RESET +
                " - %(message)s"
            )
        else:
            # Use base format if no color is specified
            colored_format = base_format 
        
        # Set the formatter with the selected format string
        formatter = logging.Formatter(colored_format)
        return formatter.format(record)

# Function to set up the logger with colored output
def setup_logger(name=__name__, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)

        # Create stream handler
        console_handler = logging.StreamHandler()

        # Apply colored formatter to the stream handler
        colored_formatter = ColoredFormatter()
        console_handler.setFormatter(colored_formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger

# Add method for custom log level
def log_llm(self, message, *args, **kws):
    if self.isEnabledFor(LLM_LOG_LEVEL):
        self._log(LLM_LOG_LEVEL, message, args, **kws)

logging.Logger.llm = log_llm

# Simple logger access
def get_logger(name=__name__):
    return setup_logger(name)

# Usage in a script
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.llm("This is a llm message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is critical.")