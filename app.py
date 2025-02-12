from fastapi import FastAPI, File, UploadFile,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil,os
from fastapi.responses import JSONResponse,FileResponse,HTMLResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import json
import uuid


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Loading YOLO model
model = torch.hub.load(r'yolov5', 'custom', path=r'yolov5s.pt', source='local')


# Endpoint to serve the static HTML file
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return FileResponse("static/index.html")

output_json_dir = Path(__file__) / '..' / 'output_json'
output_json_dir = output_json_dir.resolve()  
output_json_dir.mkdir(parents=True, exist_ok=True)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Generating a unique identifier for each request to avoid overwriting files
        unique_id = str(uuid.uuid4())

        # Reading image file
        image_data = await file.read()
        img = Image.open(BytesIO(image_data))

        # Converting to numpy array for model processing
        img_np = np.array(img)
        
        results = model(img_np)
       
        #  output path JSON       
        output_json_path = output_json_dir / f"{unique_id}.json"
        print("json Path:",output_json_path)

        # Saving the image with bounding boxes
        results.save()

        # Retrieving bounding box details
        detections = []
        for *box, conf, cls in results.xywh[0]:
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            label_id = int(cls.item())
            confidence = conf.item()
            label_name = model.names[label_id]  # Get label name from model's class names

            # Save detection information for JSON output
            detections.append({
                "class_id": label_id,
                "class_name": label_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

        # Saving  JSON detection output
        with open(output_json_path, "w") as json_file:
            json.dump({"detections": detections}, json_file, indent=4)
        print(str(output_json_path))

        return JSONResponse(content={   
            "message": "Detection completed",
            "unique_id": unique_id,
            "json_path": str(output_json_path),
            "detections": detections
        })
       

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
@app.get("/download/{unique_id}/")
    
async def download_file(unique_id: str,background_tasks: BackgroundTasks):

    try:
            
            folder_path = Path('runs/detect').resolve()
          
           
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]   
            
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
          
            directory = Path(os.path.join(folder_path,latest_subfolder))    
           
            files = os.listdir(directory)
            latest_file = files[0]
            
            
            
    
            imfile = os.path.join(folder_path,latest_subfolder, latest_file)
            # print("imfile:",str(imfile))
            json_path = output_json_dir/f"{unique_id}.json"

        # temporary folder to store the files
            temp_folder = Path(f"temp_{unique_id}")
            temp_folder.mkdir(parents=True, exist_ok=True)

            # Copy the files to the temporary folder
            shutil.copy(imfile, temp_folder / f"{unique_id}.jpg")
            
            shutil.copy(json_path, temp_folder / f"{unique_id}.json")
            
            # Zip the folder
            zip_file_path = Path("yoloDetection.zip")
            shutil.make_archive(str(zip_file_path).replace(".zip", ""), 'zip', temp_folder)

            # Clearing up the temporary folder 
            shutil.rmtree(temp_folder)
      
            if os.path.exists(os.path.join(folder_path, latest_subfolder)):
                    shutil.rmtree(Path(os.path.join(folder_path,latest_subfolder)))

            
            background_tasks.add_task(delete_file, zip_file_path)
            if not Path(zip_file_path).exists():
                return JSONResponse(content={"error": "File not found"}, status_code=404)
            response = FileResponse(zip_file_path, media_type='application/zip', filename=zip_file_path.name)
            response.headers["x-content-type-options"] = "nosniff"
            return response
            
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
def delete_file(file_path: Path):
    try:
        if file_path.exists():
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)