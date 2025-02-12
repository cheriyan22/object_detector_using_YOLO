FROM python:3.9-slim
# Setting the working directory in the container
WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./static /app/static
RUN apt-get update && apt-get install -y git
# Clone the YOLOv5 repository into the container
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5
COPY ./output_json /app/output_json
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Installing FastAPI and Uvicorn
RUN pip install fastapi uvicorn
# Copy YOLOv5  weights, and FastAPI app into the container
COPY yolov5s.pt /app/yolov5s.pt
COPY app.py /app/app.py
# Expose the port FastAPI will run on
EXPOSE 8000
# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
