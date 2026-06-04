__all__ = ["BboxLoss", "DFLoss", "YOLODetectionLoss"]


def __getattr__(name):
    if name == "BboxLoss":
        from .bbox import BboxLoss
        return BboxLoss
    if name == "DFLoss":
        from .dfl import DFLoss
        return DFLoss
    if name == "YOLODetectionLoss":
        from .yolo_detection import YOLODetectionLoss
        return YOLODetectionLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
