# =====================================================================
# 7. إعدادات الـ Cache مع معامل التنبؤ
# =====================================================================

CACHE_CONFIG = {
    "max_size": 128,                        # أقصى عدد عناصر في الـ Cache
    "ttl_seconds": 30.0,                    # مدة حياة العنصر (Time To Live)
    "prediction_coefficient": 0.75,         # معامل التنبؤ (0.0 → 1.0)
    "enable_predictive_loading": True,      # تفعيل التحميل التنبؤي
    "prefetch_threshold": 0.6,              # عتبة التنبؤ لبدء التحميل المسبق
}


# =====================================================================
# 8. حلقة المحركات الثلاثة بإشراف merge_data
# =====================================================================

ENGINE_LOOP_CONFIG = {
    "supervisor": "merge_data",             # المشرف الرئيسي على الحلقة
    "engines": [
        {
            "name": "merge_engine",
            "function": "merge_data",
            "role": "supervisor",
            "description": "يدمج البيانات ويقرر المسار التالي"
        },
        {
            "name": "cpu_engine",
            "function": "filter_for_cpu",
            "role": "worker",
            "description": "يجهز البيانات للـ CPU ويعدم نسخة GPU"
        },
        {
            "name": "gpu_engine",
            "function": "filter_for_gpu",
            "role": "worker",
            "description": "يجهز البيانات للـ GPU ويعدم نسخة CPU"
        }
    ],
    "loop_policy": {
        "max_cycles": 8,                    # أقصى عدد دورات في الحلقة
        "stop_on_stable": True,             # توقف عند استقرار الملكية
        "allow_remerge": True,              # السماح بإعادة الدمج أثناء الحلقة
        "prediction_coefficient": 0.75      # نفس معامل التنبؤ المستخدم في الـ Cache
    }
}


# =====================================================================
# 9. سياسة اتخاذ القرار داخل الحلقة (Decision Policy)
# =====================================================================

def decide_next_engine(
    current_ownership: "DataOwnership",
    prediction_score: float,
    prediction_coefficient: float = 0.75
) -> str:
    """
    يقرر المحرك التالي بناءً على حالة الملكية الحالية + درجة التنبؤ.
    
    Returns:
        "filter_cpu" | "filter_gpu" | "merge_data" | "end"
    """
    if prediction_score >= prediction_coefficient:
        # التنبؤ عالي → نفضل المسار المتوقع
        if current_ownership.name == "CPU_ALIVE":
            return "filter_gpu"
        elif current_ownership.name == "GPU_ALIVE":
            return "filter_cpu"
    
    # المسار الافتراضي حسب الملكية الحالية
    if current_ownership.name == "MERGED":
        return "filter_cpu"          # أو filter_gpu حسب السياسة
    elif current_ownership.name == "CPU_ALIVE":
        return "filter_gpu"
    elif current_ownership.name == "GPU_ALIVE":
        return "filter_cpu"
    else:
        return "merge_data"