from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    data_dir: Path = Path("dataset")
    models_dir: Path = Path("models-storage")
    output_dir: Path = Path("output")

config = Config()
