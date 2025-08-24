#!/usr/bin/env python3
"""
Pi Camera YOLO Inference Script
Optimized for 320x240 Pi camera input with 320x320 ONNX model
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from pathlib import Path

class PiYOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize the YOLO detector for Pi Camera
        
        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names (matching your training)
        self.class_names = {
            0: 'bird', 1: 'dog', 2: 'cat', 
            3: 'motorcycle', 4: 'car', 5: 'truck'
        }
        
        # Colors for each class (BGR format)
        self.colors = {
            0: (0, 255, 255),    # bird - yellow
            1: (255, 0, 0),      # dog - blue  
            2: (0, 255, 0),      # cat - green
            3: (255, 0, 255),    # motorcycle - magenta
            4: (0, 0, 255),      # car - red
            5: (255, 255, 0)     # truck - cyan
        }
        
        print(f"🤖 Loading ONNX model: {model_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']  # Pi doesn't have GPU
        )
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]  # Should be 320
        self.input_width = self.input_shape[3]   # Should be 320
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Expected input: {self.input_width}x{self.input_height}")
        print(f"   Available providers: {ort.get_available_providers()}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess Pi camera frame (320x240) for 320x320 ONNX model
        """
        original_height, original_width = frame.shape[:2]
        
        # Resize to 320x320 (optimal for your model and camera)
        input_frame = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Normalize to 0-1
        input_frame = input_frame.astype(np.float32) / 255.0
        
        # Change from HWC to CHW format
        input_frame = np.transpose(input_frame, (2, 0, 1))
        
        # Add batch dimension
        input_frame = np.expand_dims(input_frame, axis=0)
        
        return input_frame, (original_width, original_height)
    
    def postprocess_detections(self, outputs, original_size):
        """
        Process ONNX model outputs and apply NMS
        """
        original_width, original_height = original_size
        
        # Extract predictions
        predictions = outputs[0][0]  # Remove batch dimension
        
        boxes = []
        scores = []
        class_ids = []
        
        # Process each detection
        for detection in predictions.T:  # Transpose to iterate over detections
            # YOLO output format: [x_center, y_center, width, height, conf, class_scores...]
            x_center, y_center, width, height = detection[:4]
            confidence = detection[4]
            
            if confidence > self.conf_threshold:
                # Get class with highest score
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                if class_conf > self.conf_threshold:
                    # Convert from model coordinates (320x320) to original coordinates
                    x_center *= original_width / self.input_width
                    y_center *= original_height / self.input_height
                    width *= original_width / self.input_width
                    height *= original_height / self.input_height
                    
                    # Convert center format to corner format
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence * class_conf))
                    class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            class_ids = np.array(class_ids)
            
            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), 
                self.conf_threshold, self.iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                return boxes[indices], scores[indices], class_ids[indices]
        
        return [], [], []
    
    def draw_detections(self, frame, boxes, scores, class_ids):
        """
        Draw detection boxes and labels on frame
        """
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            
            # Get class info
            class_name = self.class_names.get(class_id, f'Class_{class_id}')
            color = self.colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f'{class_name}: {score:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def detect_frame(self, frame):
        """
        Run detection on a single frame
        """
        # Preprocess
        input_frame, original_size = self.preprocess_frame(frame)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_frame})
        inference_time = time.time() - start_time
        
        # Postprocess
        boxes, scores, class_ids = self.postprocess_detections(outputs, original_size)
        
        return boxes, scores, class_ids, inference_time

def main():
    parser = argparse.ArgumentParser(description='Pi Camera YOLO Detection')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.4,
                       help='IoU threshold for NMS (default: 0.4)')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video to file')
    parser.add_argument('--fps', type=int, default=10,
                       help='Target FPS for processing (default: 10)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PiYOLODetector(args.model, args.conf, args.iou)
    
    # Initialize camera
    print(f"📹 Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    # Set camera to 320x240 (native Pi camera resolution you mentioned)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Video writer setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (320, 240))
    
    print(f"🚀 Starting detection loop...")
    print(f"   Target FPS: {args.fps}")
    print(f"   Press 'q' to quit")
    
    frame_count = 0
    total_inference_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Run detection
            boxes, scores, class_ids, inference_time = detector.detect_frame(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, boxes, scores, class_ids)
            
            # Add performance info
            fps = 1.0 / inference_time if inference_time > 0 else 0
            cv2.putText(frame, f'FPS: {fps:.1f} | Detections: {len(boxes)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Pi YOLO Detection', frame)
            
            # Save video if requested
            if video_writer:
                video_writer.write(frame)
            
            # Update statistics
            frame_count += 1
            total_inference_time += inference_time
            
            if frame_count % 30 == 0:  # Print stats every 30 frames
                avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                print(f"📊 Frame {frame_count}: Avg FPS: {avg_fps:.1f}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        if frame_count > 0:
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            print(f"\\n📈 Final Statistics:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Average inference time: {total_inference_time/frame_count*1000:.1f}ms")

if __name__ == "__main__":
    main()