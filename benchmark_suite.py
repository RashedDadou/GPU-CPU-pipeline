# benchmark_suite.py

"""
ZYX Memory Pipeline Comprehensive Benchmark Suite
================================================
Measures performance across latency, memory, GPU util, cache hit rate, and throughput
"""

import os
import json
import time
import numpy as np
import torch
import psutil
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DATA STRUCTURES & ENUMS
# ============================================================================

class BenchmarkType(Enum):
    """Types of benchmarks to run"""
    LATENCY = "latency"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    GPU_UTIL = "gpu_util"
    COMPARATIVE = "comparative"


@dataclass
class MemoryMetrics:
    """Memory usage statistics"""
    peak_gpu_mb: float = 0.0
    avg_gpu_mb: float = 0.0
    peak_cpu_mb: float = 0.0
    avg_cpu_mb: float = 0.0
    memory_saved_percent: float = 0.0
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


@dataclass
class LatencyMetrics:
    """Latency statistics"""
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


@dataclass
class ThroughputMetrics:
    """Throughput statistics (items/sec)"""
    items_per_sec: float = 0.0
    mb_per_sec: float = 0.0
    samples_processed: int = 0
    total_time_sec: float = 0.0
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    total_hits: int = 0
    total_misses: int = 0
    eviction_count: int = 0
    avg_entry_age_sec: float = 0.0
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


@dataclass
class GPUUtilMetrics:
    """GPU utilization metrics"""
    avg_util_percent: float = 0.0
    peak_util_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_util_percent: float = 0.0
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    name: str
    benchmark_type: str
    data_size_mb: float
    batch_size: int
    num_iterations: int
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    cache: CacheMetrics = field(default_factory=CacheMetrics)
    gpu_util: GPUUtilMetrics = field(default_factory=GPUUtilMetrics)
    speedup_vs_baseline: float = 1.0
    memory_saved_vs_baseline: float = 0.0
    cache_improvement_vs_baseline: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# 2. RESOURCE MONITORING
# ============================================================================

class ResourceMonitor:
    """Real-time resource monitoring (CPU, GPU, Memory)"""
    
    def __init__(self, interval_sec: float = 0.01):
        self.interval = interval_sec
        self.running = False
        self.metrics_log: List[Dict] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        
    def start(self):
        """Start monitoring resources"""
        self.running = True
        self.metrics_log.clear()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.running:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': self.process.cpu_percent(),
                    'rss_mb': self.process.memory_info().rss / (1024 ** 2),
                }
                
                if torch.cuda.is_available():
                    metrics['gpu_memory_used_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                    metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
                
                self.metrics_log.append(metrics)
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                break
    
    def stop(self) -> Dict:
        """Stop monitoring and return aggregated metrics"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.metrics_log:
            return {}
        
        # Aggregate metrics
        log_array = np.array([
            [m.get('cpu_percent', 0), 
             m.get('rss_mb', 0), 
             m.get('gpu_memory_used_mb', 0),
             m.get('gpu_memory_reserved_mb', 0)]
            for m in self.metrics_log
        ])
        
        return {
            'cpu_avg_percent': float(np.mean(log_array[:, 0])),
            'cpu_peak_percent': float(np.max(log_array[:, 0])),
            'memory_avg_mb': float(np.mean(log_array[:, 1])),
            'memory_peak_mb': float(np.max(log_array[:, 1])),
            'gpu_mem_used_avg_mb': float(np.mean(log_array[:, 2])),
            'gpu_mem_used_peak_mb': float(np.max(log_array[:, 2])),
            'gpu_mem_reserved_avg_mb': float(np.mean(log_array[:, 3])),
        }


# ============================================================================
# 3. CACHE SIMULATOR
# ============================================================================

class SimpleLRUCache:
    """Simple LRU Cache for simulation"""
    
    def __init__(self, max_size_mb: float = 128.0, ttl_sec: float = 30.0):
        self.max_size_mb = max_size_mb
        self.ttl_sec = ttl_sec
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve from cache"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        data, timestamp = self.cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.ttl_sec:
            del self.cache[key]
            self.misses += 1
            return None
        
        # Update LRU order
        self.access_order.remove(key)
        self.access_order.append(key)
        self.hits += 1
        return data
    
    def put(self, key: str, data: np.ndarray):
        """Store in cache"""
        data_size_mb = data.nbytes / (1024 ** 2)
        
        # Evict if necessary
        while self._current_size_mb() + data_size_mb > self.max_size_mb and self.cache:
            evicted_key = self.access_order.pop(0)
            del self.cache[evicted_key]
            self.evictions += 1
        
        self.cache[key] = (data, time.time())
        self.access_order.append(key)
    
    def _current_size_mb(self) -> float:
        """Get current cache size in MB"""
        return sum(v[0].nbytes / (1024 ** 2) for v in self.cache.values())
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        ages = [time.time() - timestamp for _, timestamp in self.cache.values()]
        avg_age = np.mean(ages) if ages else 0.0
        
        return CacheMetrics(
            hit_rate=hit_rate,
            miss_rate=1 - hit_rate,
            total_hits=self.hits,
            total_misses=self.misses,
            eviction_count=self.evictions,
            avg_entry_age_sec=avg_age
        )


# ============================================================================
# 4. BENCHMARK SCENARIOS
# ============================================================================

class BenchmarkScenarios:
    """Collection of benchmark test scenarios"""
    
    @staticmethod
    def scenario_data_transfer(
        size_mb: int = 100,
        iterations: int = 100,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, List[float]]:
        """CPU ↔ GPU data transfer benchmark"""
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = 'cpu'
        
        data = np.random.randn(size_mb * 256 * 1024 // 8).astype(np.float32)
        times = []
        
        # Warmup
        for _ in range(10):
            tensor = torch.from_numpy(data).to(device)
            if device != 'cpu':
                torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            
            if device != 'cpu':
                tensor = torch.from_numpy(data).to(device)
                torch.cuda.synchronize()
            else:
                tensor = torch.from_numpy(data)
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        return data, times
    
    @staticmethod
    def scenario_cache_performance(
        data_size_mb: int = 100,
        num_requests: int = 10000,
        cache_size_mb: int = 128,
        access_pattern: str = 'zipf'  # 'uniform', 'zipf', 'sequential'
    ) -> Tuple[SimpleLRUCache, List[str]]:
        """Cache hit rate benchmark"""
        
        cache = SimpleLRUCache(max_size_mb=cache_size_mb)
        
        # Generate synthetic data
        num_keys = max(10, data_size_mb // 10)
        keys = [f"key_{i}" for i in range(num_keys)]
        data = {k: np.random.randn(1024 * 128).astype(np.float32) for k in keys}
        
        # Preload some keys
        for k in keys[:num_keys // 2]:
            cache.put(k, data[k])
        
        # Generate access pattern
        if access_pattern == 'zipf':
            # Zipf distribution (80/20 rule)
            weights = np.array([1 / (i + 1) for i in range(num_keys)])
            weights /= weights.sum()
            access_seq = np.random.choice(keys, size=num_requests, p=weights)
        elif access_pattern == 'sequential':
            access_seq = [keys[i % len(keys)] for i in range(num_requests)]
        else:  # uniform
            access_seq = np.random.choice(keys, size=num_requests)
        
        # Execute access pattern
        for key in access_seq:
            if cache.get(key) is None:
                cache.put(key, data[key])
        
        return cache, access_seq
    
    @staticmethod
    def scenario_gpu_computation(
        matrix_size: int = 1024,
        iterations: int = 100,
        operation: str = 'matmul'  # 'matmul', 'conv2d', 'fft'
    ) -> List[float]:
        """GPU computation benchmark"""
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return []
        
        times = []
        
        if operation == 'matmul':
            A = torch.randn(matrix_size, matrix_size, device='cuda')
            B = torch.randn(matrix_size, matrix_size, device='cuda')
            
            # Warmup
            for _ in range(10):
                torch.matmul(A, B)
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                torch.matmul(A, B)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
        
        elif operation == 'conv2d':
            x = torch.randn(16, 3, 224, 224, device='cuda')
            conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).cuda()
            
            # Warmup
            for _ in range(10):
                conv(x)
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                conv(x)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
        
        return times


# ============================================================================
# 5. BENCHMARK EXECUTOR
# ============================================================================

class BenchmarkExecutor:
    """Main benchmark executor"""
    
    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def run_latency_benchmark(
        self,
        name: str,
        test_fn: Callable,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> LatencyMetrics:
        """Run latency benchmark"""
        
        logger.info(f"🔵 Running latency benchmark: {name}")
        
        # Warmup
        for _ in range(warmup_iterations):
            test_fn()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times_ms = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            test_fn()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times_ms.append(elapsed)
        
        times_array = np.array(times_ms)
        
        metrics = LatencyMetrics(
            mean_ms=float(np.mean(times_array)),
            median_ms=float(np.median(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            std_ms=float(np.std(times_array)),
            p95_ms=float(np.percentile(times_array, 95)),
            p99_ms=float(np.percentile(times_array, 99))
        )
        
        logger.info(f"   Mean: {metrics.mean_ms:.3f}ms | Median: {metrics.median_ms:.3f}ms | P99: {metrics.p99_ms:.3f}ms")
        return metrics
    
    def run_memory_benchmark(
        self,
        name: str,
        test_fn: Callable,
        num_iterations: int = 10
    ) -> MemoryMetrics:
        """Run memory usage benchmark"""
        
        logger.info(f"🟢 Running memory benchmark: {name}")
        
        monitor = ResourceMonitor(interval_sec=0.01)
        monitor.start()
        
        for _ in range(num_iterations):
            test_fn()
        
        monitored = monitor.stop()
        
        metrics = MemoryMetrics(
            peak_gpu_mb=monitored.get('gpu_mem_used_peak_mb', 0.0),
            avg_gpu_mb=monitored.get('gpu_mem_used_avg_mb', 0.0),
            peak_cpu_mb=monitored.get('memory_peak_mb', 0.0),
            avg_cpu_mb=monitored.get('memory_avg_mb', 0.0)
        )
        
        logger.info(f"   GPU Peak: {metrics.peak_gpu_mb:.2f}MB | CPU Peak: {metrics.peak_cpu_mb:.2f}MB")
        return metrics
    
    def run_throughput_benchmark(
        self,
        name: str,
        test_fn: Callable[[], Tuple[int, float]],  # returns (samples_processed, data_size_mb)
        num_iterations: int = 50
    ) -> ThroughputMetrics:
        """Run throughput benchmark"""
        
        logger.info(f"🟡 Running throughput benchmark: {name}")
        
        total_samples = 0
        total_size_mb = 0.0
        
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            samples, size_mb = test_fn()
            total_samples += samples
            total_size_mb += size_mb
        
        elapsed_sec = time.perf_counter() - start_time
        
        metrics = ThroughputMetrics(
            items_per_sec=total_samples / elapsed_sec,
            mb_per_sec=total_size_mb / elapsed_sec,
            samples_processed=total_samples,
            total_time_sec=elapsed_sec
        )
        
        logger.info(f"   Throughput: {metrics.items_per_sec:.0f} items/sec | {metrics.mb_per_sec:.2f} MB/sec")
        return metrics
    
    def run_cache_benchmark(
        self,
        name: str,
        access_pattern: str = 'zipf',
        data_size_mb: int = 100,
        cache_size_mb: int = 128,
        num_requests: int = 10000
    ) -> CacheMetrics:
        """Run cache performance benchmark"""
        
        logger.info(f"🔴 Running cache benchmark: {name} (pattern: {access_pattern})")
        
        cache, _ = BenchmarkScenarios.scenario_cache_performance(
            data_size_mb=data_size_mb,
            num_requests=num_requests,
            cache_size_mb=cache_size_mb,
            access_pattern=access_pattern
        )
        
        metrics = cache.get_metrics()
        logger.info(f"   Hit Rate: {metrics.hit_rate*100:.2f}% | Evictions: {metrics.eviction_count}")
        
        return metrics
    
    def run_gpu_util_benchmark(
        self,
        name: str,
        test_fn: Callable,
        duration_sec: float = 5.0
    ) -> GPUUtilMetrics:
        """Run GPU utilization benchmark"""
        
        logger.info(f"🟣 Running GPU utilization benchmark: {name}")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return GPUUtilMetrics()
        
        monitor = ResourceMonitor(interval_sec=0.1)
        monitor.start()
        
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration_sec:
            test_fn()
        
        monitored = monitor.stop()
        
        # Estimate GPU utilization (simplified)
        peak_mem = monitored.get('gpu_mem_used_peak_mb', 0.0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) if torch.cuda.is_available() else 1.0
        
        metrics = GPUUtilMetrics(
            avg_util_percent=min(100.0, (monitored.get('gpu_mem_used_avg_mb', 0.0) / total_mem) * 100),
            peak_util_percent=min(100.0, (peak_mem / total_mem) * 100),
            memory_used_mb=peak_mem,
            memory_reserved_mb=monitored.get('gpu_mem_reserved_avg_mb', 0.0),
            memory_util_percent=min(100.0, (peak_mem / total_mem) * 100)
        )
        
        logger.info(f"   Peak Util: {metrics.peak_util_percent:.2f}% | Memory: {metrics.memory_used_mb:.2f}MB")
        return metrics
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save all results to JSON"""
        
        results_json = [asdict(r) for r in self.results]
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"✅ Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary report"""
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY REPORT".center(80))
        print("="*80 + "\n")
        
        for result in self.results:
            print(f"📊 Test: {result.name}")
            print(f"   Type: {result.benchmark_type}")
            print(f"   Data Size: {result.data_size_mb:.2f}MB | Batch: {result.batch_size} | Iterations: {result.num_iterations}")
            print(f"\n   ⏱️  Latency:")
            print(f"      Mean: {result.latency.mean_ms:.3f}ms | Median: {result.latency.median_ms:.3f}ms")
            print(f"      Min: {result.latency.min_ms:.3f}ms | Max: {result.latency.max_ms:.3f}ms")
            print(f"      P95: {result.latency.p95_ms:.3f}ms | P99: {result.latency.p99_ms:.3f}ms")
            
            print(f"\n   💾 Memory:")
            print(f"      GPU Peak: {result.memory.peak_gpu_mb:.2f}MB | Avg: {result.memory.avg_gpu_mb:.2f}MB")
            print(f"      CPU Peak: {result.memory.peak_cpu_mb:.2f}MB | Avg: {result.memory.avg_cpu_mb:.2f}MB")
            
            print(f"\n   ⚡ Throughput:")
            print(f"      {result.throughput.items_per_sec:.0f} items/sec | {result.throughput.mb_per_sec:.2f} MB/sec")
            
            print(f"\n   🎯 Cache:")
            print(f"      Hit Rate: {result.cache.hit_rate*100:.2f}% | Evictions: {result.cache.eviction_count}")
            
            print(f"\n   🚀 GPU Util:")
            print(f"      Peak: {result.gpu_util.peak_util_percent:.2f}% | Avg: {result.gpu_util.avg_util_percent:.2f}%")
            
            print(f"\n   📈 Improvements:")
            print(f"      Speedup: {result.speedup_vs_baseline:.2f}x | Memory Saved: {result.memory_saved_vs_baseline:.2f}%")
            print(f"      Cache Improvement: {result.cache_improvement_vs_baseline:.2f}%")
            print("\n" + "-"*80 + "\n")
