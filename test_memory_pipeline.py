# أضف: test_memory_pipeline.py

import pytest
import numpy as np

class TestMemoryPipeline:
    
    @pytest.fixture
    def pipeline(self):
        return MemoryPipeline()
    
    def test_merge_basic(self, pipeline):
        """اختبار دمج بسيط"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        
        packet = pipeline.merge_data(arr1, arr2)
        
        assert packet["ownership"] == DataOwnership.MERGED
        assert len(packet["data"]) == 2
        assert "source_0" in packet["data"]
    
    def test_filter_cpu_kills_gpu(self, pipeline):
        """اختبار أن filter_cpu يقتل بيانات GPU"""
        arr = np.array([1, 2, 3])
        packet = pipeline.merge_data(arr)
        
        # محاكاة بيانات GPU
        if TORCH_AVAILABLE:
            packet["ownership"] = DataOwnership.GPU_ALIVE
            packet["data"]["gpu_tensor"] = torch.tensor([1, 2, 3])
            
            result = pipeline.filter_for_cpu(packet)
            assert result["ownership"] == DataOwnership.CPU_ALIVE
            assert "GPU:gpu_tensor" in result["killed"]
    
    def test_dead_packet_error(self, pipeline):
        """اختبار رفع exception للبيانات الميتة"""
        packet: DataPacket = {
            "data": {},
            "ownership": DataOwnership.DEAD,
            "device": "cpu",
            "dtype": None,
            "metadata": {},
            "killed": [],
            "timestamp": time.time()
        }
        
        with pytest.raises(ValueError, match="DEAD"):
            pipeline.filter_for_cpu(packet)
    
    def test_context_cleanup(self, pipeline):
        """اختبار التنظيف التلقائي"""
        arr = np.array([1, 2, 3])
        
        with PipelineContext(pipeline) as ctx:
            packet = pipeline.merge_data(arr)
            ctx.add_packet(packet)
        
        # تأكد أن جميع الحزم أصبحت DEAD
        assert packet["ownership"] == DataOwnership.DEAD
