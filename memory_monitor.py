# أضف: zyx_memory_monitor.py

import psutil
import time
from dataclasses import dataclass

@dataclass
class MemoryStats:
    """إحصائيات الذاكرة الحية"""
    cpu_used_mb: float
    gpu_used_mb: float
    timestamp: float
    packet_count: int

class MemoryMonitor:
    """مراقب الذاكرة والأداء"""
    
    def __init__(self):
        self.stats_history: List[MemoryStats] = []
        self.process = psutil.Process()
    
    def get_current_stats(self, packet_count: int = 0) -> MemoryStats:
        """الحصول على إحصائيات الذاكرة الحالية"""
        cpu_mb = self.process.memory_info().rss / 1024 / 1024
        
        gpu_mb = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        stats = MemoryStats(
            cpu_used_mb=cpu_mb,
            gpu_used_mb=gpu_mb,
            timestamp=time.time(),
            packet_count=packet_count
        )
        self.stats_history.append(stats)
        return stats
    
    def log_report(self) -> str:
        """تقرير شامل عن استخدام الذاكرة"""
        if not self.stats_history:
            return "No data"
        
        latest = self.stats_history[-1]
        max_cpu = max(s.cpu_used_mb for s in self.stats_history)
        max_gpu = max(s.gpu_used_mb for s in self.stats_history)
        
        report = f"""
        ╔═══════════════════════════════════════╗
        ║      📊 Memory Report               ║
        ╠═══════════════════════════════════════╣
        ║ CPU Current:  {latest.cpu_used_mb:>20.2f} MB
        ║ CPU Peak:     {max_cpu:>20.2f} MB
        ║ GPU Current:  {latest.gpu_used_mb:>20.2f} MB
        ║ GPU Peak:     {max_gpu:>20.2f} MB
        ║ Packets:      {latest.packet_count:>20}
        ╚═══════════════════════════════════════╝
        """
        return report
