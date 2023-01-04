# Idea
Implement a component that can detect objects within images.\
The following is a rough and not complete list of bits and pieces to (hopefully) achieve this:
- Run a model like EfficientDet (https://github.com/google/automl/tree/master/efficientdet) for object detection
- Possibly use a model runtime / environment like Nvidia Triton
- Provide a gRPC API for communication
- Package this component into a container (K3s?)

# Services
- Inference
  - Input: Image (e.g. base64-coded)
  - Output: List of detected objects (identified by class and bounding box within the image)
- Status / Monitoring?
- Control? 