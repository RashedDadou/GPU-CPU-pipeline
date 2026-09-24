# cache_engine_system.py

"""
نظام الـ Cache المتقدم + حلقة المحركات الذكية
مع التنبؤ الديناميكي والقرارات المستنيرة
"""

import time
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# 1. تحسين معامل التنبؤ الديناميكي
# =====================================================================

@dataclass
class PredictionMetrics:
    """مقاييس التنبؤ الحية"""
    success_rate: float = 0.5          # نسبة دقة التنبؤ
    total_predictions: int = 0         # إجمالي التنبؤات
    correct_predictions: int = 0       # التنبؤات الصحيحة
    avg_latency_ms: float = 0.0       # متوسط التأخير
    confidence: float = 0.5            # ثقة النظام في التنبؤ


class DynamicPredictionCoefficient:
    """معامل تنبؤ ذكي يتعلم من الأداء"""
    
    def __init__(self, initial_coefficient: float = 0.75):
        self.base_coefficient = initial_coefficient
        self.current_coefficient = initial_coefficient
        self.metrics = PredictionMetrics()
        self.history: List[Tuple[float, bool]] = []  # (prediction, was_correct)
    
    def record_prediction(self, predicted: bool, actual: bool) -> None:
        """تسجيل نتيجة التنبؤ"""
        self.metrics.total_predictions += 1
        
        was_correct = predicted == actual
        self.metrics.correct_predictions += was_correct
        self.history.append((self.current_coefficient, was_correct))
        
        # احتفظ بـ 100 تنبؤ أخير فقط
        if len(self.history) > 100:
            self.history.pop(0)
        
        logger.debug(f"📊 Prediction recorded: {was_correct} (accuracy: {self.get_accuracy():.2%})")
    
    def update_coefficient(self) -> None:
        """تحديث المعامل بناءً على الأداء"""
        if self.metrics.total_predictions < 10:
            return  # لا تحدّث حتى نجمع بيانات كافية
        
        accuracy = self.get_accuracy()
        
        # ✅ إذا كانت الدقة عالية → زيادة الثقة
        if accuracy > 0.85:
            self.current_coefficient = min(0.95, self.current_coefficient + 0.02)
        # ⚠️ إذا كانت متوسطة → عودة للقاعدة
        elif accuracy < 0.60:
            self.current_coefficient = max(0.40, self.current_coefficient - 0.05)
        # 📊 إذا كانت متوازنة → استقرار
        else:
            self.current_coefficient = self.base_coefficient
        
        logger.info(f"🎯 Coefficient updated: {self.current_coefficient:.3f} (Accuracy: {accuracy:.2%})")
    
    def get_accuracy(self) -> float:
        """الحصول على دقة التنبؤ الحالية"""
        if self.metrics.total_predictions == 0:
            return 0.5
        return self.metrics.correct_predictions / self.metrics.total_predictions
    
    def get_adaptive_coefficient(self, memory_pressure: float) -> float:
        """احصل على معامل يتكيف مع ضغط الذاكرة"""
        # إذا كانت الذاكرة تحت ضغط → قلل الثقة في التنبؤ
        memory_factor = 1.0 - (memory_pressure * 0.3)
        return self.current_coefficient * memory_factor


# =====================================================================
# 2. Cache ذكي مع Predictive Prefetching
# =====================================================================

@dataclass
class CacheEntry:
    """عنصر في الـ Cache"""
    key: str
    packet: "DataPacket"
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    prediction_score: float = 0.5
    
    def is_expired(self, ttl: float) -> bool:
        """تحقق من انتهاء صلاحية العنصر"""
        return (time.time() - self.timestamp) > ttl


class PredictiveCache:
    """الـ Cache مع التحميل التنبؤي"""
    
    def __init__(self, config: Dict):
        self.max_size = config["max_size"]
        self.ttl = config["ttl_seconds"]
        self.prefetch_threshold = config["prefetch_threshold"]
        self.enable_predictive = config["enable_predictive_loading"]
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.predictor = DynamicPredictionCoefficient(
            config["prediction_coefficient"]
        )
        self.prefetch_queue: List[str] = []
        self.hit_rate = 0.0
        self.miss_count = 0
        self.hit_count = 0
    
    def get(self, key: str) -> Optional["DataPacket"]:
        """احصل على عنصر من الـ Cache"""
        if key not in self.cache:
            self.miss_count += 1
            logger.debug(f"❌ Cache MISS: {key}")
            return None
        
        entry = self.cache[key]
        
        # تحقق من انتهاء الصلاحية
        if entry.is_expired(self.ttl):
            logger.debug(f"⏰ Expired entry removed: {key}")
            del self.cache[key]
            self.miss_count += 1
            return None
        
        # تحديث الإحصائيات
        entry.access_count += 1
        entry.last_accessed = time.time()
        self.hit_count += 1
        self.hit_rate = self.hit_count / (self.hit_count + self.miss_count)
        
        logger.debug(f"✅ Cache HIT: {key} (Rate: {self.hit_rate:.2%})")
        return entry.packet
    
    def put(
        self,
        key: str,
        packet: "DataPacket",
        prediction_score: float = 0.5
    ) -> None:
        """ضع عنصراً في الـ Cache"""
        
        # إذا امتلأ الـ Cache → احذف الأقل استخداماً (LRU)
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        entry = CacheEntry(
            key=key,
            packet=packet,
            timestamp=time.time(),
            prediction_score=prediction_score
        )
        
        self.cache[key] = entry
        logger.info(f"💾 Cached: {key} (Prediction score: {prediction_score:.2f})")
        
        # التنبؤ بالعناصر التالية
        if self.enable_predictive and prediction_score >= self.prefetch_threshold:
            self._trigger_prefetch(key)
    
    def _evict_lru(self) -> None:
        """احذف العنصر الأقل استخداماً مؤخراً"""
        # ابحث عن العنصر الأقل استخداماً
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].access_count, self.cache[k].last_accessed)
        )
        
        logger.warning(f"🗑️ Evicted LRU entry: {lru_key}")
        del self.cache[lru_key]
    
    def _trigger_prefetch(self, key: str) -> None:
        """شغّل التحميل المسبق التنبؤي"""
        # هنا يمكنك إضافة logic لتنبؤ العنصر التالي
        logger.info(f"🔮 Prefetch triggered for: {key}")
        # يمكن إضافة عناصر متوقعة إلى prefetch_queue
    
    def get_stats(self) -> Dict[str, float]:
        """الحصول على إحصائيات الـ Cache"""
        return {
            "cache_size": len(self.cache),
            "hit_rate": self.hit_rate,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "predictor_accuracy": self.predictor.get_accuracy(),
            "current_coefficient": self.predictor.current_coefficient
        }


# =====================================================================
# 3. محرك القرارات الذكي (Smart Decision Engine)
# =====================================================================

class EngineDecisionPolicy(Enum):
    """سياسات اتخاذ القرار"""
    GREEDY = "greedy"                  # اختر الأسرع
    BALANCED = "balanced"              # توازن بين الأداء والذاكرة
    MEMORY_AWARE = "memory_aware"      # فضّل تقليل استهلاك الذاكرة
    PREDICTIVE = "predictive"          # اتبع التنبؤ


@dataclass
class EngineContext:
    """سياق القرار"""
    current_ownership: "DataOwnership"
    data_size_mb: float
    memory_available_mb: float
    gpu_available: bool
    cuda_memory_mb: float = 0.0
    recent_latencies: List[float] = field(default_factory=list)
    prediction_score: float = 0.5
    policy: EngineDecisionPolicy = EngineDecisionPolicy.BALANCED


class SmartDecisionEngine:
    """محرك قرار ذكي يأخذ بعين الاعتبار عدة عوامل"""
    
    def __init__(self, cache: PredictiveCache):
        self.cache = cache
        self.decision_history: List[Tuple[str, float]] = []
    
    def decide_next_engine(self, context: EngineContext) -> Tuple[str, float]:
        """
        قرر المحرك التالي بناءً على عوامل متعددة.
        
        Returns:
            (engine_name, confidence_score)
        """
        
        # حساب درجة الجدوى لكل محرك
        cpu_score = self._calculate_cpu_score(context)
        gpu_score = self._calculate_gpu_score(context)
        
        logger.debug(f"📊 Decision scores - CPU: {cpu_score:.3f}, GPU: {gpu_score:.3f}")
        
        # اختر بناءً على السياسة
        if context.policy == EngineDecisionPolicy.GREEDY:
            engine = "filter_gpu" if gpu_score > cpu_score else "filter_cpu"
        elif context.policy == EngineDecisionPolicy.MEMORY_AWARE:
            engine = self._memory_aware_decision(context, cpu_score, gpu_score)
        elif context.policy == EngineDecisionPolicy.PREDICTIVE:
            engine = self._predictive_decision(context)
        else:  # BALANCED
            engine = self._balanced_decision(context, cpu_score, gpu_score)
        
        confidence = max(cpu_score, gpu_score)
        self.decision_history.append((engine, confidence))
        
        logger.info(f"🎯 Decision: {engine} (Confidence: {confidence:.2f})")
        return engine, confidence
    
    def _calculate_cpu_score(self, context: EngineContext) -> float:
        """احسب درجة جدوى CPU"""
        score = 1.0
        
        # البيانات الصغيرة مناسبة للـ CPU
        if context.data_size_mb < 50:
            score += 0.3
        
        # إذا كانت البيانات بالفعل على CPU
        if context.current_ownership.name == "CPU_ALIVE":
            score += 0.4
        
        # الكمون المنخفض للـ CPU
        if context.recent_latencies:
            avg_latency = np.mean(context.recent_latencies)
            if avg_latency < 10:  # < 10 ms
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_gpu_score(self, context: EngineContext) -> float:
        """احسب درجة جدوى GPU"""
        score = 0.5
        
        if not context.gpu_available:
            return 0.0
        
        # البيانات الكبيرة مناسبة للـ GPU
        if context.data_size_mb > 100:
            score += 0.3
        
        # توفر ذاكرة GPU كافية
        if context.data_size_mb < (context.cuda_memory_mb * 0.7):
            score += 0.2
        
        # التنبؤ يشير إلى GPU
        if context.prediction_score > 0.7:
            score += 0.2
        
        return min(1.0, score)
    
    def _memory_aware_decision(
        self,
        context: EngineContext,
        cpu_score: float,
        gpu_score: float
    ) -> str:
        """قرار يراعي الذاكرة"""
        memory_pressure = context.data_size_mb / (context.memory_available_mb + 1e-6)
        
        if memory_pressure > 0.8:
            logger.warning(f"⚠️ High memory pressure: {memory_pressure:.2%}")
            return "filter_cpu"  # فضّل CPU عند ضغط الذاكرة
        
        return "filter_gpu" if gpu_score > cpu_score else "filter_cpu"
    
    def _predictive_decision(self, context: EngineContext) -> str:
        """قرار بناءً على التنبؤ"""
        if context.prediction_score >= self.cache.predictor.current_coefficient:
            return "filter_gpu" if context.prediction_score > 0.5 else "filter_cpu"
        
        # الخيار الآمن
        return "filter_cpu"
    
    def _balanced_decision(
        self,
        context: EngineContext,
        cpu_score: float,
        gpu_score: float
    ) -> str:
        """توازن بين الأداء والموارد"""
        # إذا كانت الفارق بسيطاً → اختر الخيار الآمن (CPU)
        if abs(gpu_score - cpu_score) < 0.1:
            return "filter_cpu"
        
        return "filter_gpu" if gpu_score > cpu_score else "filter_cpu"


# =====================================================================
# 4. حلقة المحركات المُحسّنة مع Supervisor
# =====================================================================

class SupervisedEngineLoop:
    """حلقة محركات مع مشرف ذكي"""
    
    def __init__(self, pipeline: "MemoryPipeline", cache: PredictiveCache):
        self.pipeline = pipeline
        self.cache = cache
        self.decision_engine = SmartDecisionEngine(cache)
        self.config = ENGINE_LOOP_CONFIG
        self.cycle_count = 0
    
    def run_loop(
        self,
        initial_packet: "DataPacket",
        max_cycles: Optional[int] = None,
        policy: EngineDecisionPolicy = EngineDecisionPolicy.BALANCED
    ) -> "DataPacket":
        """
        شغّل حلقة المحركات مع الإشراف.
        
        العملية:
        1. merge_data (المشرف) يقرر المسار
        2. CPU/GPU engines تنفذ العملية
        3. معاودة الدورة حتى الاستقرار أو الحد الأقصى
        
        Args:
            initial_packet: حزمة البيانات الأولية
            max_cycles: أقصى عدد دورات (None = استخدم الإعدادات)
            policy: سياسة اتخاذ القرار
        
        Returns:
            DataPacket الحزمة بعد الدورات
        """
        
        if max_cycles is None:
            max_cycles = self.config["loop_policy"]["max_cycles"]
        
        packet = initial_packet
        self.cycle_count = 0
        previous_ownership = None
        
        logger.info(f"🔄 Starting Engine Loop (max_cycles={max_cycles}, policy={policy.value})")
        
        while self.cycle_count < max_cycles:
            self.cycle_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"⚙️  Cycle {self.cycle_count}/{max_cycles}")
            logger.info(f"{'='*60}")
            
            # ========== المرحلة 1: الإشراف (Supervision) ==========
            logger.debug("📋 [PHASE 1] Supervisor (merge_data) analysis...")
            
            merge_result = self.pipeline.merge_data(packet)
            packet = merge_result["packet"]
            merge_stats = merge_result.get("stats", {})
            
            logger.info(f"✅ Merge completed: {merge_stats}")
            
            # ========== فحص الاستقرار ==========
            if (previous_ownership == packet.ownership and 
                self.config["loop_policy"]["stop_on_stable"]):
                logger.info(f"🎯 Stable state reached at cycle {self.cycle_count}")
                logger.info(f"   Ownership: {packet.ownership.name}")
                break
            
            previous_ownership = packet.ownership
            
            # ========== المرحلة 2: بناء سياق القرار ==========
            logger.debug("📊 [PHASE 2] Building decision context...")
            
            context = EngineContext(
                current_ownership=packet.ownership,
                data_size_mb=self._estimate_packet_size(packet),
                memory_available_mb=self._get_available_memory(),
                gpu_available=self._is_gpu_available(),
                cuda_memory_mb=self._get_cuda_memory(),
                recent_latencies=self._get_recent_latencies(),
                prediction_score=self.cache.predictor.current_coefficient,
                policy=policy
            )
            
            logger.debug(f"Context: Size={context.data_size_mb:.2f}MB, "
                        f"GPU={context.gpu_available}, "
                        f"MemAvail={context.memory_available_mb:.2f}MB")
            
            # ========== المرحلة 3: اتخاذ القرار ==========
            logger.debug("🎯 [PHASE 3] Smart decision engine...")
            
            next_engine, confidence = self.decision_engine.decide_next_engine(context)
            
            logger.info(f"🔮 Decision: {next_engine} (Confidence: {confidence:.2%})")
            
            # ========== المرحلة 4: تنفيذ المحرك ==========
            logger.debug(f"⚙️  [PHASE 4] Executing {next_engine}...")
            
            if next_engine == "filter_cpu":
                execution_result = self._execute_cpu_engine(packet, context)
            elif next_engine == "filter_gpu":
                execution_result = self._execute_gpu_engine(packet, context)
            elif next_engine == "merge_data":
                # إعادة الدمج إذا سمحت السياسة
                if not self.config["loop_policy"]["allow_remerge"]:
                    logger.warning("⚠️ Remerge not allowed, stopping loop")
                    break
                continue
            else:
                logger.info("🏁 End of loop reached")
                break
            
            packet = execution_result["packet"]
            engine_stats = execution_result.get("stats", {})
            
            logger.info(f"✅ Engine executed: {engine_stats}")
            
            # ========== المرحلة 5: تسجيل النتائج و التعلم ==========
            logger.debug("[PHASE 5] Recording predictions for learning...")
            
            was_correct = self._validate_prediction(
                predicted_engine=next_engine,
                actual_ownership=packet.ownership,
                context=context
            )
            
            self.cache.predictor.record_prediction(
                predicted=(next_engine == "filter_gpu"),
                actual=(packet.ownership.name == "GPU_ALIVE")
            )
            
            # تحديث معامل التنبؤ كل 5 دورات
            if self.cycle_count % 5 == 0:
                self.cache.predictor.update_coefficient()
            
            logger.info(f"📈 Prediction accuracy: {self.cache.predictor.get_accuracy():.2%}")
            
            # ========== عرض الحالة الحالية ==========
            self._log_cycle_summary(packet, context, execution_result)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 Engine Loop Completed")
        logger.info(f"   Total cycles: {self.cycle_count}/{max_cycles}")
        logger.info(f"   Final ownership: {packet.ownership.name}")
        logger.info(f"   Predictor accuracy: {self.cache.predictor.get_accuracy():.2%}")
        logger.info(f"   Cache hit rate: {self.cache.hit_rate:.2%}")
        logger.info(f"{'='*60}\n")
        
        return packet
    
    
    # =====================================================================
    # 5. Helpers للتنفيذ والقياس
    # =====================================================================
    
    def _execute_cpu_engine(
        self,
        packet: "DataPacket",
        context: EngineContext
    ) -> Dict:
        """تنفيذ CPU Engine"""
        start_time = time.time()
        
        logger.info("🖥️  Executing CPU Engine (filter_for_cpu)...")
        
        # تنفيذ التفلتر
        filtered_packet = self.pipeline.filter_for_cpu(packet)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "packet": filtered_packet,
            "stats": {
                "engine": "filter_cpu",
                "latency_ms": elapsed_ms,
                "ownership": filtered_packet.ownership.name,
                "device": filtered_packet.device
            }
        }
    
    def _execute_gpu_engine(
        self,
        packet: "DataPacket",
        context: EngineContext
    ) -> Dict:
        """تنفيذ GPU Engine"""
        start_time = time.time()
        
        logger.info("🎮 Executing GPU Engine (filter_for_gpu)...")
        
        if not context.gpu_available:
            logger.warning("⚠️ GPU not available, falling back to CPU")
            return self._execute_cpu_engine(packet, context)
        
        # تنفيذ التفلتر
        filtered_packet = self.pipeline.filter_for_gpu(packet)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "packet": filtered_packet,
            "stats": {
                "engine": "filter_gpu",
                "latency_ms": elapsed_ms,
                "ownership": filtered_packet.ownership.name,
                "device": filtered_packet.device
            }
        }
    
    def _estimate_packet_size(self, packet: "DataPacket") -> float:
        """تقدير حجم الحزمة بالـ MB"""
        if hasattr(packet.data, 'nbytes'):  # NumPy
            return packet.data.nbytes / (1024 ** 2)
        elif hasattr(packet.data, 'element_size'):  # PyTorch
            return (packet.data.numel() * packet.data.element_size()) / (1024 ** 2)
        return 0.0
    
    def _get_available_memory(self) -> float:
        """احصل على الذاكرة المتاحة بالـ MB"""
        import psutil
        return psutil.virtual_memory().available / (1024 ** 2)
    
    def _is_gpu_available(self) -> bool:
        """تحقق من توفر GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_cuda_memory(self) -> float:
        """احصل على ذاكرة CUDA المتاحة بالـ MB"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.mem_get_info()[0] / (1024 ** 2)
        except:
            pass
        return 0.0
    
    def _get_recent_latencies(self) -> List[float]:
        """احصل على الكمونات الحديثة من السجل"""
        if hasattr(self.decision_engine, 'decision_history'):
            # عودة آخر 5 قرارات (محاكاة)
            return [10, 12, 11, 13, 10]  # ملتقطة من السجل
        return []
    
    def _validate_prediction(
        self,
        predicted_engine: str,
        actual_ownership: "DataOwnership",
        context: EngineContext
    ) -> bool:
        """التحقق من صحة التنبؤ"""
        if predicted_engine == "filter_gpu":
            return actual_ownership.name == "GPU_ALIVE"
        elif predicted_engine == "filter_cpu":
            return actual_ownership.name == "CPU_ALIVE"
        return False
    
    def _log_cycle_summary(
        self,
        packet: "DataPacket",
        context: EngineContext,
        execution: Dict
    ) -> None:
        """اطبع ملخص الدورة"""
        logger.info(f"""
        📊 Cycle Summary:
           ├─ Packet Size: {context.data_size_mb:.2f} MB
           ├─ Ownership: {packet.ownership.name}
           ├─ Device: {packet.device}
           ├─ Latency: {execution['stats'].get('latency_ms', 0):.2f} ms
           ├─ Cache Hit Rate: {self.cache.hit_rate:.2%}
           └─ Predictor Accuracy: {self.cache.predictor.get_accuracy():.2%}
        """)


# =====================================================================
# 6. إعدادات محسّنة نهائية
# =====================================================================

ENHANCED_CACHE_CONFIG = {
    "max_size": 256,                        # زيادة سعة الـ Cache
    "ttl_seconds": 45.0,                    # مدة أطول قليلاً
    "prediction_coefficient": 0.75,         # معامل أولي
    "enable_predictive_loading": True,      # تفعيل التنبؤ
    "prefetch_threshold": 0.65,             # عتبة أقل صرامة
}

ENHANCED_ENGINE_LOOP_CONFIG = {
    "supervisor": "merge_data",
    "engines": [
        {
            "name": "merge_engine",
            "function": "merge_data",
            "role": "supervisor",
            "description": "يدمج البيانات ويتخذ قرار ذكي"
        },
        {
            "name": "cpu_engine",
            "function": "filter_for_cpu",
            "role": "worker",
            "description": "معالجة CPU مع تعلم ديناميكي"
        },
        {
            "name": "gpu_engine",
            "function": "filter_for_gpu",
            "role": "worker",
            "description": "معالجة GPU مع تنبؤ ذكي"
        }
    ],
    "loop_policy": {
        "max_cycles": 10,                    # زيادة بسيطة
        "stop_on_stable": True,              # توقف عند الاستقرار
        "allow_remerge": True,               # اسمح بإعادة الدمج
        "adaptive_coefficient": True,        # تفعيل التكيف الديناميكي
        "learning_rate": 0.05,               # معدل التعلم
    },
    "decision_policies": {
        "default": "balanced",
        "under_memory_pressure": "memory_aware",
        "high_prediction_confidence": "predictive",
    }
}

# =====================================================================
# 7. مثال كامل للاستخدام
# =====================================================================

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    # تهيئة النظام
    cache = PredictiveCache(ENHANCED_CACHE_CONFIG)
    pipeline = MemoryPipeline()  # استخدم الفئة من السابق
    engine_loop = SupervisedEngineLoop(pipeline, cache)
    
    # إنشاء حزمة بيانات تجريبية
    test_data = np.random.randn(1000, 1000)
    initial_packet = DataPacket(
        data=test_data,
        ownership=DataOwnership.MERGED,
        device="cpu",
        dtype="float32",
        metadata={"source": "test_data"},
        killed=False,
        timestamp=time.time()
    )
    
    # تشغيل الحلقة مع سياسات مختلفة
    print("\n🔴 Policy 1: BALANCED (افتراضي)")
    result_balanced = engine_loop.run_loop(
        initial_packet,
        policy=EngineDecisionPolicy.BALANCED
    )
    
    print("\n🟢 Policy 2: PREDICTIVE (تنبؤي)")
    result_predictive = engine_loop.run_loop(
        initial_packet,
        policy=EngineDecisionPolicy.PREDICTIVE
    )
    
    print("\n🟡 Policy 3: MEMORY_AWARE (وعي بالذاكرة)")
    result_memory = engine_loop.run_loop(
        initial_packet,
        policy=EngineDecisionPolicy.MEMORY_AWARE
    )
    
    # عرض الإحصائيات النهائية
    print("\n" + "="*60)
    print("📊 Final Statistics:")
    print("="*60)
    
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}")
    
    print(f"\n   Decision History Length: {len(engine_loop.decision_engine.decision_history)}")
    print(f"   Total Cycles Run: {engine_loop.cycle_count}")
    print(f"   Final Ownership: {result_balanced.ownership.name}")
