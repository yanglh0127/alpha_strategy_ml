import logging
from logging.handlers import TimedRotatingFileHandler
# logging.basicConfig(format='%(asctime)s %(levelname)s [%(filename)s->%(funcName)s:%(lineno)d]\t%(message)s',
#                     level=logging.DEBUG)

def init_logger(name="main", echo=True, filename=None, level=None, fmt=None):
    fmt = fmt or '%(asctime)s %(levelname)s [%(filename)s->%(funcName)s:%(lineno)d]\t%(message)s'
    formatter = logging.Formatter(fmt) #设置日志格式
    logger = logging.getLogger(name)
    if level:
        if level == "DEBUG" or level == logging.DEBUG:
            logger.setLevel(level = logging.DEBUG)
        elif level == "INFO" or level == logging.INFO:
            logger.setLevel(level = logging.INFO)
        elif level == "WARN" or level == "WARNING" or level == logging.WARN:
            logger.setLevel(level = logging.WARN)
        elif level == "FATAL" or level == logging.FATAL:
            logger.setLevel(level = logging.FATAL)
        else:
            logger.setLevel(level = level)
    if echo:
        console_handler = logging.StreamHandler() # 控制台Handler
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    if filename:
        file_handler = TimedRotatingFileHandler(filename=filename, when='MIDNIGHT', interval=1, backupCount=30, encoding='utf-8')
        # file_handler = logging.FileHandler("log.txt") # 文件Handler
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    logger.info(f"logger({name or ''}) initialized with format: {fmt}")
    return logger

def get_logger(name=None):
    return logging.getLogger(name)

logger = get_logger(name="main")
