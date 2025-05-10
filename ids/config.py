# ids/config.py
from pathlib import Path
from typing import List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    """Глобальные настройки проекта."""

    data_root:    Path = Path("data")
    results_root: Path = Path("results")
    seed:           int = 42

    # packet | uniflow | biflow
    mode: str = "biflow"


    models: List[str] = ["lr", "rf", "mlp"]

    cv: Union[int, str] = 0

    # DL
    epochs:      int = 20
    batch_size:  int = 64

    model_config = SettingsConfigDict(
        env_prefix="MQTTIDS_",
        env_file=".env",
        extra="ignore",        
    )

cfg = Config()       
