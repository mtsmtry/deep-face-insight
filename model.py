import onnx
from onnx import helper, shape_inference
import onnxoptimizer
from onnxruntime.quantization import quantize_dynamic, QuantType


# Load the ONNX model
model = onnx.load('models/inswapper_128.onnx')

model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = -1
model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = -1

#model = onnxoptimizer.optimize(model)

# Check model
onnx.checker.check_model(model)


# Apply shape inference on the model
#inferred_model = shape_inference.infer_shapes(model)

# Save the updated model
onnx.save(model, 'models/inswapper_128_batch.onnx')
