import logging
import pathlib

loggers = {}

def setup_logging(module_name):

    if module_name in loggers:
        
        return loggers[module_name]
    
    log_directory = pathlib.Path('logs')
    log_directory.mkdir(exist_ok=True)
    
    log_file = log_directory / f'{module_name}.log'

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, mode='w')
    
    # formatter
    console_logging_format = logging.Formatter(fmt='%(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    file_logging_format = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    
    # Set formatters for handlers
    console_handler.setFormatter(console_logging_format)
    file_handler.setFormatter(file_logging_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    loggers[module_name] = logger
    
    return logger
