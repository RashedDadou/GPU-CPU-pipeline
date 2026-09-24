# تحسين الملف الأصلي:

from typing import Literal, Union, Callable
from enum import Enum

# ✅ أنواع بيانات أقوى
DeviceType = Literal["cpu", "cuda", "mps"]
TorchDtype = Literal["float32", "float64", "int32", "int64"]

class PipelineStage(Enum):
    """مراحل Pipeline محددة بدقة"""
    MERGE = "merge"
    FILTER_CPU = "filter_cpu"
    FILTER_GPU = "filter_gpu"
    COMPUTE_CPU = "compute_cpu"
    COMPUTE_GPU = "compute_gpu"
    END = "end"

# ✅ Callback للعمليات المخصصة
ProcessCallback = Callable[[DataPacket], DataPacket]
