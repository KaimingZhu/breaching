"""
@Encoding:      UTF-8 
@File:          dlg_cifar10.py

@Introduction:  a same executable script as `.ipynb` file in `breaching/examples`, see @Reference for detail info.
@Author:        Kaiming Zhu
@Date:          2024/3/22 16:56
@Reference:     ./examples/Deep Leakage from Gradients - Optimization-based Attack - ConvNet CIFAR-10.ipynb
"""

try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os
    os.chdir("..")
    import breaching

import torch
import logging, sys

# Initialize Logger
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

# setup attack config
cfg = breaching.get_config(overrides=["case=1_single_image_small", "attack=deepleakage"])
device = torch.device(f'cuda:1') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
print(setup)

# modify config option
cfg.case.data.partition = "unique-class"
cfg.case.user.user_idx = 1
cfg.case.user.provide_labels = False

# Initialize all entities
user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

# make FL simulation
server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)
user.plot(true_user_data)

# attack
reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

# evalute attack metrics
metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                    server.model, order_batch=True, compute_full_iip=False,
                                    cfg_case=cfg.case, setup=setup)

# show reconstructed image
user.plot(reconstructed_user_data)