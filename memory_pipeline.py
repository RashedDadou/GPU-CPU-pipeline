# أضف ملف جديد: zyx_memory_pipeline.py

import time
import numpy as np
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MemoryPipeline:
    """التطبيق الفعلي لمسار الذاكرة"""
    
    def __init__(self, auto_cleanup: bool = True):
        self.auto_cleanup = auto_cleanup
        self._memory_log: List[Dict[str, Any]] = []
    
    def merge_data(
        self,
        *sources: ArrayLike,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataPacket:
        """دمج البيانات من مصادر متعددة"""
        
        merged_dict = {}
        for i, source in enumerate(sources):
            key = f"source_{i}"
            # تحويل إلى NumPy بصفة موحدة
            if isinstance(source, torch.Tensor):
                merged_dict[key] = source.detach().cpu().numpy()
            elif isinstance(source, np.ndarray):
                merged_dict[key] = source.copy()
            else:
                merged_dict[key] = np.array(source)
        
        packet: DataPacket = {
            "data": merged_dict,
            "ownership": DataOwnership.MERGED,
            "device": "cpu",
            "dtype": str(merged_dict[f"source_0"].dtype),
            "metadata": metadata or {},
            "killed": [],
            "timestamp": time.time()
        }
        
        logger.info(f"✅ Merged {len(sources)} sources → {len(merged_dict)} keys")
        return packet
    
    def filter_for_cpu(
        self,
        packet: DataPacket,
        required_keys: Optional[List[str]] = None
    ) -> DataPacket:
        """فلترة وتحضير البيانات للـ CPU"""
        
        # ✅ التحقق من الصحة
        self._validate_packet(packet)
        if packet["ownership"] == DataOwnership.DEAD:
            raise ValueError("❌ لا يمكن فلترة بيانات DEAD!")
        
        # 🔪 إعدام بيانات GPU إن وجدت
        if packet["ownership"] == DataOwnership.GPU_ALIVE:
            self._kill_gpu_data(packet)
        
        # 📋 تصفية المفاتيح المطلوبة
        filtered_data = {}
        keys_to_process = required_keys or list(packet["data"].keys())
        
        for key in keys_to_process:
            if key in packet["data"]:
                data = packet["data"][key]
                # تحويل إلى NumPy بكفاءة
                if isinstance(data, torch.Tensor):
                    filtered_data[key] = data.detach().cpu().numpy()
                else:
                    filtered_data[key] = data
        
        packet["data"] = filtered_data
        packet["ownership"] = DataOwnership.CPU_ALIVE
        packet["device"] = "cpu"
        packet["timestamp"] = time.time()
        
        logger.info(f"✅ Filtered to CPU: {len(filtered_data)} arrays ready")
        return packet
    
    def filter_for_gpu(
        self,
        packet: DataPacket,
        device: str = "cuda",
        required_keys: Optional[List[str]] = None,
        dtype: Optional[Any] = None
    ) -> DataPacket:
        """فلترة وتحضير البيانات للـ GPU"""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("❌ PyTorch غير مثبت!")
        
        # ✅ التحقق من الصحة
        self._validate_packet(packet)
        if packet["ownership"] == DataOwnership.DEAD:
            raise ValueError("❌ لا يمكن فلترة بيانات DEAD!")
        
        # 🔪 إعدام بيانات CPU إن وجدت
        if packet["ownership"] == DataOwnership.CPU_ALIVE:
            self._kill_cpu_data(packet)
        
        # تحديد نوع البيانات
        torch_dtype = dtype or torch.float32
        
        # 📋 تصفية وتحويل
        filtered_data = {}
        keys_to_process = required_keys or list(packet["data"].keys())
        
        for key in keys_to_process:
            if key in packet["data"]:
                data = packet["data"][key]
                # تحويل إلى Tensor
                if isinstance(data, torch.Tensor):
                    tensor = data.to(device=device, dtype=torch_dtype)
                else:
                    tensor = torch.from_numpy(np.asarray(data)).to(
                        device=device, dtype=torch_dtype
                    )
                filtered_data[key] = tensor
        
        packet["data"] = filtered_data
        packet["ownership"] = DataOwnership.GPU_ALIVE
        packet["device"] = device
        packet["dtype"] = str(torch_dtype)
        packet["timestamp"] = time.time()
        
        logger.info(f"✅ Filtered to GPU ({device}): {len(filtered_data)} tensors ready")
        return packet
    
    # === Private Helper Methods ===
    
    def _validate_packet(self, packet: DataPacket) -> None:
        """التحقق من سلامة الحزمة"""
        required_keys = {"data", "ownership", "device", "metadata", "killed", "timestamp"}
        if not all(k in packet for k in required_keys):
            raise ValueError(f"❌ Packet غير صحيح! Missing: {required_keys - set(packet.keys())}")
        
        if not isinstance(packet["ownership"], DataOwnership):
            raise TypeError(f"❌ ownership يجب أن يكون DataOwnership enum!")
    
    def _kill_gpu_data(self, packet: DataPacket) -> None:
        """إعدام بيانات GPU وتحرير الذاكرة"""
        if TORCH_AVAILABLE:
            for key, data in packet["data"].items():
                if isinstance(data, torch.Tensor):
                    packet["killed"].append(f"GPU:{key}")
                    del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.warning(f"🔪 Killed GPU data: {packet['killed']}")
    
    def _kill_cpu_data(self, packet: DataPacket) -> None:
        """إعدام بيانات CPU والمصفوفات الكبيرة"""
        import gc
        for key, data in packet["data"].items():
            if isinstance(data, np.ndarray) and data.nbytes > 1e7:  # > 10 MB
                packet["killed"].append(f"CPU:{key}")
                del data
        gc.collect()
        logger.warning(f"🔪 Killed CPU data: {packet['killed']}")
        
        
class PipelineContext:
    """Context Manager للتعامل الآمن مع البيانات"""
    
    def __init__(self, pipeline: MemoryPipeline):
        self.pipeline = pipeline
        self.packets: List[DataPacket] = []
    
    def __enter__(self):
        logger.info("📂 Opening Memory Pipeline Context...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """تنظيف تلقائي عند الخروج"""
        logger.info("🧹 Cleaning up Memory Pipeline...")
        
        for packet in self.packets:
            # إعدام جميع البيانات
            if packet["ownership"] == DataOwnership.GPU_ALIVE:
                self.pipeline._kill_gpu_data(packet)
            elif packet["ownership"] == DataOwnership.CPU_ALIVE:
                self.pipeline._kill_cpu_data(packet)
            
            packet["ownership"] = DataOwnership.DEAD
        
        self.packets.clear()
        logger.info("✅ Pipeline cleaned successfully")
    
    def add_packet(self, packet: DataPacket) -> None:
        """تسجيل حزمة للتنظيف التلقائي"""
        self.packets.append(packet)

# ✅ الاستخدام:
with PipelineContext(pipeline) as ctx:
    packet = pipeline.merge_data(array1, array2)
    ctx.add_packet(packet)
    
    cpu_packet = pipeline.filter_for_cpu(packet)
    ctx.add_packet(cpu_packet)
    # التنظيف التلقائي عند الخروج

