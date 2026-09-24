# أضف: zyx_pipeline_dashboard.py

class PipelineDashboard:
    """لوحة التحكم والتصور"""
    
    def __init__(self, pipeline: MemoryPipeline):
        self.pipeline = pipeline
        self.monitor = MemoryMonitor()
        self.optimizer = PipelineOptimizer()
    
    def print_status(self) -> None:
        """طباعة حالة Pipeline الحالية"""
        stats = self.monitor.get_current_stats()
        
        status = f"""
        ╔════════════════════════════════════════════════════════╗
        ║           🔧 MEMORY PIPELINE STATUS                   ║
        ╠════════════════════════════════════════════════════════╣
        ║                                                        ║
        ║ 🖥️  CPU Memory:   {stats.cpu_used_mb:>12.2f} MB                      
        ║ 🎮 GPU Memory:   {stats.gpu_used_mb:>12.2f} MB                      
        ║ 📦 Active Packets: {stats.packet_count:>9}                        
        ║ ⏰ Timestamp:     {time.strftime('%H:%M:%S', time.localtime(stats.timestamp))}              
        ║                                                        ║
        ║ Status: ✅ OPERATIONAL                                ║
        ║                                                        ║
        ╚════════════════════════════════════════════════════════╝
        """
        print(status)
    
    def generate_report(self) -> str:
        """تقرير شامل"""
        memory_report = self.monitor.log_report()
        
        report = f"""
        
        {'='*60}
        COMPREHENSIVE PIPELINE REPORT
        {'='*60}
        
        {memory_report}
        
        Device Configuration:
        ├─ PyTorch Available: {TORCH_AVAILABLE}
        ├─ CUDA Available: {torch.cuda.is_available() if TORCH_AVAILABLE else 'N/A'}
        └─ Primary Device: {'cuda:0' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'}
        
        {'='*60}
        """
        return report


# ✅ استخدام شامل:
if __name__ == "__main__":
    # تهيئة
    pipeline = MemoryPipeline(auto_cleanup=True)
    dashboard = PipelineDashboard(pipeline)
    batch_processor = BatchPipeline(pipeline, batch_size=5)
    
    # بيانات تجريبية
    data_arrays = [
        np.random.rand(1000, 1000) for _ in range(3)
    ]
    
    # معالجة باستخدام Context
    with PipelineContext(pipeline) as ctx:
        print("🚀 Starting Pipeline Processing...\n")
        
        # دمج البيانات
        packet = pipeline.merge_data(*data_arrays, metadata={"source": "test"})
        ctx.add_packet(packet)
        
        # قياس الأداء
        benchmark = PipelineOptimizer.suggest_device(
            sum(arr.nbytes for arr in data_arrays) / 1e6
        )
        print(f"📊 Suggested Device: {benchmark}")
        
        # معالجة CPU
        cpu_packet = pipeline.filter_for_cpu(packet)
        ctx.add_packet(cpu_packet)
        
        # معالجة GPU إن توفرت
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_packet = pipeline.filter_for_gpu(packet)
            ctx.add_packet(gpu_packet)
        
        # عرض الحالة
        dashboard.print_status()
    
    # التقرير النهائي
    print(dashboard.generate_report())
