import os
from pathlib import Path


def make_dir(path:str)->None:
    """
    @param path: it creates directory with given string
    @return: None
    """
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)



def make_paths(observation_name:str,start_time:str)-> tuple:
    """
    @param config: it contains data and model configuration parameters. To change param check config.json
    @return: when u run main.py, it creates directory with that date under observation directory
    """

    saved_models_path=Path.cwd() / "saved_models"
    saved_models_path.mkdir(exist_ok=True)

    observation_path= saved_models_path / observation_name

    if not os.path.exists(saved_models_path):
        saved_models_path.mkdir(exist_ok=True,parents=True)

    if not os.path.exists(observation_path):
        observation_path.mkdir(exist_ok=True)

    time_path=observation_path / start_time
    time_path.mkdir(exist_ok=True)

    data_path=time_path / "data"
    data_path.mkdir(exist_ok=True)

    models_dir=time_path/ "models"
    models_dir.mkdir(exist_ok=True)


    return time_path,data_path,models_dir


def create_start_time():
    import datetime
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return start_time

def get_model_class(problem_type:str,module_name:str,model_name:str):
    import  importlib

    imported_module=importlib.import_module(f"model.{problem_type}.{module_name}")
    model=getattr(imported_module,model_name)
    return model



