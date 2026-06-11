from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    data_dir: Path = Path("dataset")
    models_dir: Path = Path("models")
    output_dir: Path = Path("output")

config = Config()
