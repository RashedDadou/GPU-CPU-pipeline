# أضف: zyx_memory_advanced.py

class PipelineOptimizer:
    """محسّن الأداء الذكي"""
    
    @staticmethod
    def suggest_device(data_size_mb: float) -> DeviceType:
        """اقتراح الجهاز الأفضل بناءً على حجم البيانات"""
        if data_size_mb < 50:  # بيانات صغيرة
            return "cpu"
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            available_gpu = torch.cuda.get_device_properties(0).total_memory / 1e6
            if available_gpu > data_size_mb * 2:
                return "cuda"
        return "cpu"
    
    @staticmethod
    def optimize_pipeline_route(
        packet: DataPacket,
        cpu_heavy: bool = False,
        gpu_heavy: bool = False
    ) -> PipelineStage:
        """تحديد المسار الأمثل للمعالجة"""
        
        # حساب حجم البيانات
        total_size = sum(
            arr.nbytes if isinstance(arr, (np.ndarray, torch.Tensor)) else 0
            for arr in packet["data"].values()
        )
        total_size_mb = total_size / 1e6
        
        if gpu_heavy and TORCH_AVAILABLE and torch.cuda.is_available():
            return PipelineStage.COMPUTE_GPU
        elif cpu_heavy or total_size_mb < 50:
            return PipelineStage.COMPUTE_CPU
        else:
            return PipelineStage.COMPUTE_GPU if torch.cuda.is_available() else PipelineStage.COMPUTE_CPU


class BatchPipeline:
    """معالجة دفعات متعددة من البيانات"""
    
    def __init__(self, pipeline: MemoryPipeline, batch_size: int = 10):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.batches: List[List[DataPacket]] = []
    
    def add_to_batch(self, packet: DataPacket) -> None:
        """إضافة حزمة إلى الدفعة الحالية"""
        if not self.batches or len(self.batches[-1]) >= self.batch_size:
            self.batches.append([])
        self.batches[-1].append(packet)
        logger.info(f"📦 Added to batch. Current size: {len(self.batches[-1])}/{self.batch_size}")
    
    def process_batch(self, batch_idx: int, processor: ProcessCallback) -> List[DataPacket]:
        """معالجة دفعة محددة"""
        if batch_idx >= len(self.batches):
            raise IndexError(f"❌ Batch {batch_idx} doesn't exist")
        
        batch = self.batches[batch_idx]
        results = []
        
        logger.info(f"⚙️ Processing batch {batch_idx} with {len(batch)} packets...")
        
        for i, packet in enumerate(batch):
            try:
                result = processor(packet)
                results.append(result)
                logger.info(f"  ✅ Processed packet {i}/{len(batch)}")
            except Exception as e:
                logger.error(f"  ❌ Error processing packet {i}: {e}")
                results.append(None)
        
        return results
    
    def process_all_batches(self, processor: ProcessCallback) -> List[List[DataPacket]]:
        """معالجة جميع الدفعات"""
        all_results = []
        
        for batch_idx in range(len(self.batches)):
            results = self.process_batch(batch_idx, processor)
            all_results.append(results)
        
        logger.info(f"✅ Processed all {len(self.batches)} batches")
        return all_results


class AdaptivePipeline:
    """Pipeline تكيفي يتعلم من الأداء"""
    
    def __init__(self, pipeline: MemoryPipeline):
        self.pipeline = pipeline
        self.monitor = MemoryMonitor()
        self.performance_history: Dict[str, List[float]] = {
            "cpu_times": [],
            "gpu_times": [],
            "memory_peaks": []
        }
    
    def benchmark_device(self, packet: DataPacket, iterations: int = 5) -> Dict[str, float]:
        """قياس الأداء على CPU و GPU"""
        results = {}
        
        # قياس CPU
        times_cpu = []
        for _ in range(iterations):
            start = time.time()
            cpu_packet = self.pipeline.filter_for_cpu(packet.copy())
            elapsed = time.time() - start
            times_cpu.append(elapsed)
        
        results["cpu_avg_ms"] = np.mean(times_cpu) * 1000
        
        # قياس GPU إن توفر
        if TORCH_AVAILABLE and torch.cuda.is_available():
            times_gpu = []
            for _ in range(iterations):
                start = time.time()
                gpu_packet = self.pipeline.filter_for_gpu(packet.copy(), device="cuda")
                elapsed = time.time() - start
                times_gpu.append(elapsed)
            
            results["gpu_avg_ms"] = np.mean(times_gpu) * 1000
            results["speedup"] = results["cpu_avg_ms"] / results["gpu_avg_ms"]
        
        self.performance_history["cpu_times"].extend(times_cpu)
        return results
    
    def get_recommendation(self, packet: DataPacket) -> str:
        """توصية ذكية بناءً على السجل"""
        if not self.performance_history["cpu_times"]:
            return "⚠️ No benchmark data yet"
        
        avg_cpu = np.mean(self.performance_history["cpu_times"])
        
        if "gpu_times" in self.performance_history and self.performance_history["gpu_times"]:
            avg_gpu = np.mean(self.performance_history["gpu_times"])
            speedup = avg_cpu / avg_gpu
            
            if speedup > 1.5:
                return f"🚀 Recommend GPU (Speedup: {speedup:.2f}x)"
            else:
                return f"💻 Recommend CPU (GPU not worth overhead)"
        
        return f"💻 CPU is stable ({avg_cpu*1000:.2f} ms)"

