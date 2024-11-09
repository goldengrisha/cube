# Build and run:
1. `docker build -t triton_text_to_image .`
2. `docker run -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_text_to_image`

Or 
1. `docker-compose up`

# Optional:
1. Done
2. Apply adjustments to the latent space to improve the quality of the generated images.
In diffusion models, the latent space is an important feature that influences the quality and diversity of generated images. It can be explored through various techniques like:
Latent space regularization: Regularizing the latent space can make the generated images more consistent. This could include adjusting the variance of latent vectors or using a more structured latent space.
Latent space interpolation: It can be experimented with interpolating between different latent vectors to generate smoother transitions or novel image outputs.
Using a technique like CLIP-guided diffusion or latent diffusion models can help improve image quality by adjusting the latent space during the generation process.

3. Adjust main.py to perform inference via TensorRT or ONNX Runtime.
Switching to TensorRT or ONNX Runtime can significantly speed up inference, especially on GPUs. Here’s a basic approach for using ONNX Runtime with existing model:
```python
import onnx
import onnxruntime as ort

# Load the ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Set up the ONNX runtime session
sess = ort.InferenceSession("model.onnx")

# Prepare the input for the model
inputs = {sess.get_inputs()[0].name: input_data}

# Perform inference
outputs = sess.run(None, inputs)

# Handle the outputs (e.g., process the generated image)
```
For TensorRT, it can be used NVIDIA's trt package to convert ONNX model to a TensorRT engine and perform inference.

4. Utilize a quantized version of the model to reduce the model size and improve inference speed.
Using a quantized model can help reduce memory usage and improve inference speed, particularly on GPUs with limited resources. It can be used libraries like Hugging Face's transformers to load a quantized version of model:
```python
from transformers import AutoModelForImageGeneration

model = AutoModelForImageGeneration.from_pretrained("model-name", revision="quantized")
Ensure the quantized model is compatible with framework (like PyTorch or ONNX).
```
