from torchmetrics import Metric


class ToothF1Score(Metric):
    """Stub — training-only metric, not used during inference."""

    full_state_update = False

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        return {}
