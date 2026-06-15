"""
Exponential decay scheduler for teacher forcing ratio.
"""


class ExponentialDecayScheduler:
    """
    Exponential decay of teacher forcing ratio.

    ratio = start_ratio * (1 - decay_rate)^epoch

    Args:
        start_ratio: Initial teacher forcing ratio (default: 1.0)
        decay_rate: Decay rate per epoch (default: 0.005)
        min_ratio: Minimum ratio (default: 0.0)
    """

    def __init__(
        self,
        start_ratio: float = 1.0,
        decay_rate: float = 0.005,
        min_ratio: float = 0.0,
    ):
        self.start_ratio = start_ratio
        self.decay_rate = decay_rate
        self.min_ratio = min_ratio

    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        ratio = self.start_ratio * ((1 - self.decay_rate) ** epoch)
        return max(self.min_ratio, ratio)
