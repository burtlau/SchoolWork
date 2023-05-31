import logging
from logging.config import fileConfig
import yaml

def is_palindrome(s):
    logging.getLogger(__name__).info("is_palindrome({})".format(s))
    if s is None:
        logging.getLogger(__name__).info("is_palindrome invoked with None")
        return False
    
    if not isinstance(s, str):
        logging.getLogger(__name__).error("is_palindrome requires string parameter, passed {}".format(s))
        raise TypeError("string parameters required")

    return s == s[::-1]


if __name__ == '__main__':
    # create logger
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    #ch = logging.StreamHandler()
    #ch.setLevel(logging.DEBUG)

    # create formatter
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    #ch.setFormatter(formatter)

    # add ch to logger
    #logger.addHandler(ch)
    #fileConfig('log_config.ini')

    f = open("./log_config.yaml")
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    f.close()
    
    print(is_palindrome("abba"))
    print(is_palindrome("not a palindrome"))
    print(is_palindrome(99))