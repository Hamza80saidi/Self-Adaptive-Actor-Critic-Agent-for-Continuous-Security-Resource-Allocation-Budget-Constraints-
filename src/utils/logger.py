"""
logger.py
=========
Wrapper simple autour de tensorboard (+ export CSV optionnel) pour suivre
l'entraînement : reward par épisode, loss actor/critic, entropie, etc.

TODO Jour 3:
    [ ] Implémenter Logger.log_scalar(name, value, step)
    [ ] Implémenter Logger.close()
"""

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = "logs/"):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name: str, value: float, step: int):
        self.writer.add_scalar(name, value, step)

    def close(self):
        self.writer.close()
