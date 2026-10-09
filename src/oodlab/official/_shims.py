"""Minimal stand-ins for the two OpenOOD-VLM imports used by the vendored files.

The vendored postprocessors only need ``BasePostprocessor`` (a config holder)
and ``openood.utils.comm`` (used for progress bars). Nothing algorithmic lives
here, so the vendored code behaves exactly as in the official repository.
"""
import types


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net, id_loader_dict, ood_loader_dict):
        pass


comm = types.SimpleNamespace(
    is_main_process=lambda: True,
    get_rank=lambda: 0,
    get_world_size=lambda: 1,
    synchronize=lambda: None,
)
