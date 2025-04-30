# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .custom.humanoid_config import XBotLCfg, XBotLCfgPPO
from .custom.humanoid_env import XBotLFreeEnv
# cowa
from .cowa.cowa_config import CowaCfg, CowaCfgPPO
from .cowa.cowa_env import CowaFreeEnv

# cowa FIX
from .cowa_fix.cowa_fix_config import CowaFixCfg, CowaFixCfgPPO
from .cowa_fix.cowa_fix_env import CowaFixEnv

# cowa VAE
from .cowa_vae.cowa_vae_config import CowaCfg_VAE, CowaCfgPPO_VAE
# cowa EST
from .cowa_est.cowa_est_config import CowaCfg_EST,CowaCfgPPO_EST
# cowa RMA
from .cowa_rma.cowa_rma_config import CowaCfgPPO_RMA, CowaCfg_RMA


from humanoid.utils.task_registry import task_registry

# cowa Wheel

task_registry.register( "humanoid_ppo", XBotLFreeEnv, XBotLCfg(), XBotLCfgPPO() )
task_registry.register( "cowa", CowaFreeEnv, CowaCfg(), CowaCfgPPO() )
task_registry.register( "cowa_vae" ,CowaFreeEnv, CowaCfg_VAE(), CowaCfgPPO_VAE())
task_registry.register( "cowa_fix", CowaFixEnv, CowaFixCfg(), CowaFixCfgPPO() )
task_registry.register( "cowa_est" ,CowaFreeEnv, CowaCfg_EST(), CowaCfgPPO_EST())
task_registry.register( "cowa_rma" ,CowaFreeEnv, CowaCfg_RMA(), CowaCfgPPO_RMA())
