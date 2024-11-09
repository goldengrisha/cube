# Use Triton as the base image
FROM nvcr.io/nvidia/tritonserver:24.10-py3

WORKDIR /app

COPY ./requirements.txt /app

RUN apt install python3

# Install Python dependencies
RUN pip install -r requirements.txt

# Set environment variables
ENV MODEL_NAME="StableDiffusion_1_5"

# Run the server
CMD ["python3", "main.py"]
