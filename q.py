
import onnx
from onnx import helper, shape_inference
import onnxoptimizer
from onnxruntime.quantization import quantize_dynamic, QuantType


quantize_dynamic('models/inswapper_128_batch.onnx', 'models/inswapper_128_Batch_q.onnx', weight_type=QuantType.QInt8)
