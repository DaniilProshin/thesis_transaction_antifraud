from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from torch import nn

class Preprocessor:
    def fit(self, df):
        return self
    def transform(self, df):
        return 


class MixedAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward():
        return

@dataclass
class Detector:
    preprocessor: Preprocessor
    net: MixedAE
    forest: object
    threshold: float
    model_id: str
    config: dict

    def predict(self, df):
        result
        return result
    