# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from ppsci.loss import mtl
from ppsci.loss.base import Loss
from ppsci.loss.chamfer import ChamferLoss
from ppsci.loss.func import FunctionalLoss
from ppsci.loss.integral import IntegralLoss
from ppsci.loss.kl import KLLoss
from ppsci.loss.l1 import L1Loss
from ppsci.loss.l1 import PeriodicL1Loss
from ppsci.loss.l2 import L2Loss
from ppsci.loss.l2 import L2RelLoss
from ppsci.loss.l2 import PeriodicL2Loss
from ppsci.loss.mae import MAELoss
from ppsci.loss.mse import CausalMSELoss
from ppsci.loss.mse import MSELoss
from ppsci.loss.mse import MSELossWithL2Decay
from ppsci.loss.mse import PeriodicMSELoss

__all__ = [
    "Loss",
    "FunctionalLoss",
    "IntegralLoss",
    "L1Loss",
    "PeriodicL1Loss",
    "L2Loss",
    "L2RelLoss",
    "PeriodicL2Loss",
    "MAELoss",
    "CausalMSELoss",
    "ChamferLoss",
    "MSELoss",
    "MSELossWithL2Decay",
    "PeriodicMSELoss",
    "KLLoss",
    "mtl",
]


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (DictConfig): Loss config.

    Returns:
        Loss: Callable loss object.
    """
    cfg = copy.deepcopy(cfg)

    loss_cls = cfg.pop("name")
    loss = eval(loss_cls)(**cfg)
    return loss
