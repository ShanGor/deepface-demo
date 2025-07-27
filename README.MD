# DeepFace Demo Project
After some ID verification process, we have requirements from regulators to verify the identity of the person by eyeball checking.
We want to use a model to compare the ID doc and the selfie and provide similarity scores to facilitate the process.

After some research, we found that InsightFace has better LFW score but the license is not fit for commercial use. FaceNet is a better model but the license is fairly good and allowed for commercial use. That's why we choose DeepFace as an actively maintained front-end.

There are some cloud solutions, if you want to expose customer data to cloud providers:
- Amazon Rekognition
- Microsoft Azure Face API

### LFW (Labeled Faces in the Wild) Scores
FaceNet, VGG-Face, ArcFace and Dlib are overperforming ones based on experiments - see [`BENCHMARKS`](https://github.com/serengil/deepface/tree/master/benchmarks) for more details. You can find the measured scores of various models in DeepFace and the reported scores from their original studies in the following table.

| Model          | Measured Score | Declared Score     |
| -------------- | -------------- | ------------------ |
| Facenet512     | 98.4%          | 99.6%              |
| Human-beings   | 97.5%          | 97.5%              |
| Facenet        | 97.4%          | 99.2%              |
| Dlib           | 96.8%          | 99.3 %             |
| VGG-Face       | 96.7%          | 98.9%              |
| ArcFace        | 96.7%          | 99.5%              |
| GhostFaceNet   | 93.3%          | 99.7%              |
| SFace          | 93.0%          | 99.5%              |
| OpenFace       | 78.7%          | 92.9%              |
| DeepFace       | 69.0%          | 97.3%              |
| DeepID         | 66.5%          | 97.4%              |