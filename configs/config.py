import platform
import yaml
from pathlib import Path


def config_reader():
    new_cwd = ""
    cwd = Path.cwd()
    if platform.system() == "Linux":
        array = str(cwd).split("/")
        for i in range(len(array)):
            if array[i] == "nasa-pdm":
                new_cwd += "/" + array[i]
                break
            else:
                new_cwd += "/"+array[i]
    elif platform.system() == "Windows":
        array = str(cwd).split("\\")
        for i in range(len(array)):
            if i == 0:
                new_cwd += array[i]
            elif array[i] == "nasa-pdm":
                new_cwd += "\\" + array[i]
                break
            else:
                new_cwd += "\\" + array[i]
    else:
        array = str(cwd).split("/")
        for i in range(len(array)):
            if array[i] == "nasa-pdm":
                new_cwd += "/" + array[i]
                break
            else:
                new_cwd += "/" + array[i]
    data_folder = Path(fr"{new_cwd}/configs/config.yaml")
    print(data_folder)
    with open(data_folder, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg
def set_configuration(cfg) -> tuple:
    x = 1
    cfg1 = {}
    available_formats = []
    available_models = []
    available_problems = []
    error_list = []
    for i in cfg:
        if i != "DATA":
            available_formats.append(i)
    for i in available_formats:
        cfg1.update(cfg[i])
    for j in cfg1:
        if j != "DATA":
            available_problems.append(j)

    for cfg_id, cfg_info in cfg1.items():
        if cfg_id != "DATA":
            for key in cfg_info:
                if key not in available_models:
                    available_models.append(key)
    while True:
        print(F"{Bcolors.HEADER}Available formats are :{Bcolors.ENDC}", available_formats)
        print("----------------------------------------------------")
        print(f"{Bcolors.HEADER}Available problems are :{Bcolors.ENDC}", available_problems)
        print("----------------------------------------------------")
        print(f"{Bcolors.HEADER}Available models are :{Bcolors.ENDC}", available_models)
        for i in available_formats:
            print(f"{Bcolors.WARNING}{i:<15}{Bcolors.ENDC}     {x:>10}")
            error_list.append(str(x))
            x += 1
        x = 1
        print("----------------------------------------------------")
        format = input(f"{Bcolors.BOLD}SELECT YOUR FORMAT : {Bcolors.ENDC}")
        if format not in error_list:
            raise ValueError(f"{Bcolors.WARNING}Wrong value entered{Bcolors.ENDC}")
        for i in available_problems:
            print(f"{Bcolors.WARNING}{i:<15}{Bcolors.ENDC}     {x:>10}")
            error_list.append(str(x))
            x += 1
        x = 1
        print("----------------------------------------------------")
        problem = input(f"{Bcolors.BOLD}SELECT YOUR PROBLEM TYPE : {Bcolors.ENDC}")
        if format not in error_list:
            raise ValueError(f"{Bcolors.WARNING}Wrong value entered{Bcolors.ENDC}")

        for i in available_models:
            print(f"{Bcolors.WARNING}{i:<15}{Bcolors.ENDC}     {x:>10}")
            x += 1
        x = 1
        print("----------------------------------------------------")
        error_list.append(str(len(available_models)))
        value = input(f"{Bcolors.BOLD}SELECT YOUR MODEL  : {Bcolors.ENDC}")
        observation = input(
            f"{Bcolors.BOLD}Type what do you want to observe. Otherwise press enter to skip:{Bcolors.ENDC} ")
        value = value.split(",")
        if observation == "":
            print(f"No observation choosed, observation='code_dev' ")
            observation = 'code_dev'
        else:
            observation = f"{observation}"
        return_config = {}
        total_dict = {}
        config = cfg["DATA"][available_formats[int(format) - 1] + "_DATA"]
        for i in value:
            return_config.update(config)
            return_config.update(cfg[available_formats[int(format) - 1]][available_problems[int(problem) - 1]][
                          available_models[int(i) - 1]])
            total_dict[available_models[int(i) - 1]] = return_config
            return_config = {}
        config = total_dict
        return config, observation



def make_cfg_attr_global(cfg, available_formats, available_models):
    # ÇALIŞMASI İÇİN SET_CONFIG FONKSIYONUNDA AVAILABLE MODELS VE AVAILABLE FORMAT LISTELERİNİ DÖNDÜR
    parameters = ["batch_size", "epochs", "max_trials", "lr_list"]
    for i in available_formats:

        for j in available_models:

            for k in parameters:
                l = f"{i}_{j}_{k}"
                globals()[l] = cfg[i][j][k]

    # print(QUICK_LSTM_max_trials)
    # return QUICK_LSTM_batch_size,QUICK_LSTM_epochs,QUICK_LSTM_max_trials,QUICK_LSTM_lr_list


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
        self.BOLD = ''


class Initializer:
    def __init__(self, **acb3):
        for key, value in acb3.items():
            setattr(self, key, value)
        print(self.scaler, self.batch_size,self.epochs)


class Parser(dict):

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = Parser(value)
        return value

def flatten(dictionary):
    output = dict()
    for k, v in dictionary.items():
        if isinstance(v, dict):
            output.update(flatten(v))
        else:
            output[k] = v
    return output
def convert_to_dict(input_cfg, *args) -> dict:
    """
    @param input_cfg: girilen inputlara göre oluşan cfg
    @param args: dict formatı dışında girilen inputlar
    @return: dict formatında inputlar gelmese bile onları kullanabilmek için dönüştürülen dict
    """
    dictionary = {}
    flatten_cfg = flatten(input_cfg)
    for i in args:
        if i in input_cfg.keys():

            if not isinstance(i, dict):
                dictionary[str(i)] = input_cfg[str(i)]
            else:
                dictionary.update(i)
    dictionary.update(flatten_cfg)
    return dictionary


def test_config():
    input_cfg, observation = set_configuration(config_reader())
    # dictionary = convert_to_dict(input_cfg, "lr_list", "batch_size", "window_size")
    # print(F"Output value = {Bcolors.OKGREEN} {dictionary}{Bcolors.ENDC}")
    input_cfg= Parser(convert_to_dict(input_cfg,"batch_size", "scaler"))
    #print(input_cfg.batch_size)
    #Initializer(**convert_to_dict(input_cfg,"batch_size", "scaler","epochs"))
    #print(scaler,window_size)
    #print(scaler.LSTM.window_size)
def call_config():
    input_cfg, observation = set_configuration(config_reader())
    return input_cfg,observation
