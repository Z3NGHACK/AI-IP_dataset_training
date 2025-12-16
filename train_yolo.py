from ultralytics import YOLO

# Load a pretrained YOLOv8 model (yolov8n = nano, fast for testing; use yolov8s or m for better accuracy)
model = YOLO('yolov8n.pt')  # It will download automatically on first run

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=50,          # You can increase to 100 later
    imgsz=640,
    batch=16,           # learn 1 time 16 images
    name='car_detection',  # Results saved in runs/detect/car_detection
    patience=10,        # Early stopping if no improvement
    device=0 if __import__('torch').cuda.is_available() else 'cpu'  # Use GPU if available
)

# Validate on validation set
val_results = model.val()

# Test on test set
test_results = model.val(split='test')

# Print key metrics
print("Validation Results:")
print(f"Precision: {val_results.box.p:.3f}")
print(f"Recall: {val_results.box.r:.3f}")
print(f"mAP@0.5: {val_results.box.map50:.3f}")
print(f"mAP@0.5:0.95: {val_results.box.map:.3f}")

print("\nTest Results:")
print(test_results.box)

# Optional: Run inference on a few test images to see results
model.predict(source='Dataset/test/images', save=True, name='test_predictions')
print("Predictions saved in runs/detect/test_predictions/")