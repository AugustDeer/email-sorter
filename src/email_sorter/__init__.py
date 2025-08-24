from pathlib import Path

import torch as t

t.set_float32_matmul_precision("high")

default_output_dir = Path("./output")
