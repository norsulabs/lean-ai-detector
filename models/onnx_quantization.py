from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

onnx_model_path = "lean-ai-text-detector.onnx"
quantized_model_path = "lean-ai-text-detector-8bit.onnx"
onnx_opt_model = onnx.load(onnx_model_path)
quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)
