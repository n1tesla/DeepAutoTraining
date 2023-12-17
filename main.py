__author__="Tunahan AKYOL"

__status__="Developing"


from train.pipeline import start_search
from configs.config import call_config


def main():
    config_all,observation=call_config()
    start_search(config_all,observation)


if __name__=="__main__":
    main()