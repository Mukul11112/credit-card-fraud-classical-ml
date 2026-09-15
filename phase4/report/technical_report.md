# Technical Report

## CIFAR-10 Image Classification Using Deep Learning, Model Compression, and API Deployment

**Author:** Mukul

**Project:** AI/ML Internship Portfolio

**Technology Stack:** Python, PyTorch, scikit-learn, FastAPI, Docker, Locust

---

# 1. Abstract

This project presents an end-to-end machine learning workflow covering data preparation, deep learning, model evaluation, model optimization, deployment, and performance testing.

The deep learning component focuses on image classification using the CIFAR-10 dataset. A custom convolutional neural network (CNN) was first developed and trained from scratch to establish a baseline. A ResNet18 model was then implemented using transfer learning to improve classification performance.

The project further investigated training-data efficiency and model compression through unstructured pruning. A 20% pruning strategy was applied to the ResNet18 model, followed by fine-tuning for five epochs to recover classification performance.

The final model was integrated into a FastAPI application and containerized using Docker. The API was tested under concurrent load using Locust to evaluate reliability, latency, and throughput.

The overall workflow demonstrates the transition from model development and experimentation to a deployable machine learning service.

---

# 2. Introduction

Machine learning systems require more than model training. A practical machine learning solution must include data preparation, model development, evaluation, optimization, deployment, validation, and performance testing.

This project was developed as an end-to-end AI/ML portfolio project. The work progressed from machine learning experimentation to deep learning and finally to production-style model serving.

For image classification, the CIFAR-10 dataset was selected because it provides a standard benchmark for evaluating convolutional neural networks.

Two deep learning approaches were investigated:

1. A custom CNN trained from scratch.
2. ResNet18 using transfer learning.

The custom CNN established a baseline for classification performance. ResNet18 was then evaluated as a stronger architecture based on residual learning and transfer learning.

The project was subsequently extended through data-efficiency experiments, model pruning, fine-tuning, REST API development, Docker containerization, and Locust-based load testing.

---

# 3. Objectives

The main objectives of the project were:

- Build an end-to-end machine learning workflow.
- Perform data preprocessing and preparation.
- Develop a CNN from scratch for image classification.
- Study transfer learning using ResNet18.
- Compare deep learning model performance.
- Analyze the effect of training-data size on model performance.
- Apply model pruning for computational optimization.
- Fine-tune the pruned model.
- Compare accuracy, F1 score, model size, parameters, and inference time.
- Deploy the trained model using FastAPI.
- Containerize the application using Docker.
- Perform concurrent API load testing using Locust.
- Analyze the complete workflow from model development to deployment.

---

# 4. Dataset

## 4.1 CIFAR-10

The deep learning experiments used the CIFAR-10 dataset.

CIFAR-10 contains 60,000 colour images belonging to 10 classes. Each image has a resolution of 32 × 32 pixels.

The ten classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset consists of 50,000 training images and 10,000 test images.

The dataset was loaded using the PyTorch `torchvision.datasets.CIFAR10` interface and processed using PyTorch data loaders.

---

# 5. Data Preprocessing

The CIFAR-10 images were converted into tensors before being passed to the neural networks.

Channel-wise normalization was applied using the following values:

**Mean:**

`[0.4914, 0.4822, 0.4465]`

**Standard Deviation:**

`[0.2470, 0.2435, 0.2616]`

The preprocessing pipeline converted the raw image data into normalized tensors suitable for PyTorch models.

The training data was loaded using shuffled batches, while the test data was loaded without shuffling to provide consistent evaluation.

---

# 6. Methodology

The overall deep learning and deployment workflow consisted of the following stages:

1. Dataset loading
2. Image preprocessing
3. Custom CNN development
4. CNN training
5. CNN evaluation
6. ResNet18 transfer learning
7. ResNet18 evaluation
8. Data-efficiency experiments
9. Model pruning
10. Pruned-model fine-tuning
11. Model evaluation
12. FastAPI integration
13. Docker containerization
14. Locust load testing
15. Results analysis

The workflow was designed to evaluate not only classification accuracy but also computational efficiency and deployment performance.

---

# 7. Custom CNN

A custom convolutional neural network was developed as the initial deep learning baseline.

The model was trained from scratch without pretrained weights.

The architecture included:

- Convolutional layers
- Batch normalization
- Activation functions
- Pooling layers
- Dropout

Training used:

- Cross-entropy loss
- Weight decay
- Learning-rate scheduling
- Batch normalization
- Dropout

The purpose of the custom CNN was to establish a baseline against which the transfer-learning approach could be compared.

## 7.1 Custom CNN Results

The recorded experiment produced the following results:

| Metric | Result |
|---|---:|
| Test Accuracy | 81.62% |
| F1 Score | 81.50% |
| Training Time | 20.74 minutes |
| Parameters | 806,218 |

The custom CNN achieved useful classification performance while using considerably fewer parameters than ResNet18.

However, the experiment demonstrated that a stronger architecture could provide improved classification performance.

---

# 8. ResNet18 Transfer Learning

## 8.1 Model Selection

ResNet18 was selected as the transfer-learning architecture.

ResNet18 uses residual connections that allow information and gradients to flow through deeper neural networks more effectively.

A pretrained ResNet18 model was loaded and its final fully connected classification layer was replaced to produce predictions for the 10 CIFAR-10 classes.

The workflow was:

```text
Pretrained ResNet18
        ↓
Replace final classification layer
        ↓
10 CIFAR-10 classes
        ↓
Fine-tuning
        ↓
Evaluation

## 8.2 Transfer Learning‹‹

Transfer learning allows a model that has already learned useful visual representations from a large dataset to be adapted to a new classification task.
In this project, a pretrained ResNet18 model was adapted for CIFAR-10 image classification.
The original final classification layer was replaced with a new fully connected layer containing 10 output classes corresponding to the CIFAR-10 categories.
The pretrained model was then fine-tuned on the CIFAR-10 training dataset.
The main advantage of this approach was that the model could reuse visual features learned from the large ImageNet dataset instead of learning all image representations from scratch.

## 8.3 Training Configuration
The ResNet18 model was trained using the following configuration:
| Parameter         |                             Value |
| ----------------- | --------------------------------: |
| Model             |                          ResNet18 |
| Dataset           |                          CIFAR-10 |
| Number of Classes |                                10 |
| Optimizer         |                              Adam |
| Loss Function     |                  CrossEntropyLoss |
| Batch Size        |                                64 |
| Weight Decay      |                            0.0001 |
| Learning Rate     |                             0.001 |
| Training Approach |                 Transfer Learning |
| Device            | MPS/CPU depending on availability |

The model was trained on the CIFAR-10 training dataset and evaluated on the test dataset.

## 8.4 Evaluation Metrics
The model was evaluated using several classification metrics:
Accuracy
Precision
Recall
F1 Score
Training Time
Parameter Count
Accuracy measures the overall percentage of correctly classified images.
Precision measures the proportion of predicted samples that were correctly classified.
Recall measures the proportion of actual samples that were correctly identified.
F1 score provides a combined measure of precision and recall.

##8.5 ResNet18 Results
The recorded ResNet18 transfer-learning experiment produced the following results:
| Metric        |        Result |
| ------------- | ------------: |
| Test Accuracy |        94.36% |
| Precision     |        94.52% |
| Recall        |        94.36% |
| F1 Score      |        94.35% |
| Training Time | 33.23 minutes |
| Parameters    |    11,181,642 |

The ResNet18 model achieved 94.36% test accuracy and an F1 score of 94.35%.
Compared with the custom CNN baseline, ResNet18 provided substantially higher classification performance.
The improvement demonstrates the benefit of residual learning and transfer learning for CIFAR-10 image classification.

#9. Data-Efficiency Experiment

An additional experiment investigated how the amount of training data affected model performance and training time.
Three training-data configurations were evaluated:

Training Data	Training Samples	Validation Accuracy	Test Accuracy	Training Time
25%	11,250	93.04%	90.95%	11.41 min
50%	22,500	93.66%	93.78%	18.57 min
100%	45,000	95.08%	94.60%	33.02 min

The results show that increasing the training-data size generally improved model performance.
Using 25% of the training data reduced training time substantially, but the test accuracy was lower than the full-data configuration.
The 100% configuration achieved the highest test accuracy while requiring the longest training time.
This experiment demonstrates the trade-off between computational cost and model performance.

#10. Model Compression and Pruning

##10.1 Motivation

Deep learning models can contain millions of parameters, making them computationally expensive to execute.
Model pruning provides a method for introducing sparsity by setting selected model weights to zero.
In this project, unstructured L1 pruning was used to introduce sparsity into the ResNet18 model.'

##10.2 Pruning Method

A copy of the trained ResNet18 model was created.
L1 unstructured pruning was applied to the weights of convolutional and linear layers.
The pruning ratio was:
20%
The pruning workflow was:
Trained ResNet18
        ↓
Create model copy
        ↓
Apply 20% L1 unstructured pruning
        ↓
Fine-tune pruned model
        ↓
Evaluate
        ↓
Measure sparsity and efficiency
        ↓
Make pruning permanent

##10.3 Fine-Tuning

After applying 20% pruning, the pruned model was fine-tuned for five epochs.
The fine-tuning configuration used:

Loss function: CrossEntropyLoss
Optimizer: Adam
Learning rate: 0.0001
Weight decay: 0.0001
Fine-tuning epochs: 5
Batch size: 64
The measured fine-tuning time was:
1.96 minutes
The training accuracy increased across the five epochs:

Epoch	Training Loss	Training Accuracy
1	1.2375	57.68%
2	0.6952	79.56%
3	0.5125	82.15%
4	0.3810	86.59%
5	0.2922	89.79%

The increasing training accuracy indicates that the pruned network adapted to the modified weight structure during fine-tuning.

##10.4 Pruned Model Evaluation

The fine-tuned pruned model produced:

Metric	Result
Accuracy	   80.49%
Precision	   80.68%
Recall	           80.49%
F1 Score	   80.48%
Inference Time	   2.078 seconds
Fine-Tuning Time.  1.96 minutes
Sparsity	   20%

The pruning experiment demonstrates that the model retained substantial classification capability after introducing 20% weight sparsity.

# 11. Experimental Comparison

The original ResNet18 transfer-learning model and the pruned and fine-tuned ResNet18 model were evaluated to study the trade-off between classification performance and computational efficiency.

| Metric |           | Original ResNet18 |                 |Pruned + Fine-Tuned ResNet18 |
| Accuracy |           |94.36% |                                 |80.49% |
| F1 Score |           |94.35% |                                 |80.48% |
| Parameters |         |11,181,642 |                            |11,181,642 |
| Model Size |         |42.69 MB|                                 |42.69 MB |
| Inference Time |     |Not directly comparable|                |2.078 s |
| Training Time |       |33.23 min|                              |1.96 min |
| Sparsity |             |0%|                                     |20%|

The pruned model achieved lower classification accuracy than the original ResNet18 after pruning and fine-tuning.

The fine-tuned pruned model achieved 80.49% accuracy and an F1 score of 80.48%, while introducing approximately 20% sparsity.

The parameter count and dense model size remained approximately the same because unstructured pruning sets selected weights to zero rather than physically removing parameters from the network architecture.

The pruning experiment therefore demonstrates the trade-off between predictive performance and model sparsity.


#12. FastAPI Deployment

##12.1 API Architecture

The trained ResNet18 model was integrated into a FastAPI application.

The API provides endpoints for health checking and image prediction.

The main endpoints are:

Method	Endpoint	         Purpose
GET	/	             Basic API information
GET	/health	             Health/status check
POST	/predict	     Image classification

The prediction endpoint accepts an image through a multipart file upload.

##12.2 Prediction Workflow

The prediction workflow is:

Client
   ↓
POST /predict
   ↓
Image Upload
   ↓
Input Validation
   ↓
Image Preprocessing
   ↓
ResNet18 Model
   ↓
Class Prediction
   ↓
Confidence Calculation
   ↓
JSON Response

The prediction response contains the predicted class and confidence score.

Example response:
{
    "class_name": "cat",
    "confidence": 0.6637
}

##12.3 Input Validation

The API performs validation before sending the image to the model.

The validation process handles:

1.Missing file
2.Empty file
3.Invalid image
4.Unsupported image content
5.Invalid prediction input

This prevents malformed requests from directly reaching the model.

##12.4 Error Handling

Error handling was implemented to provide meaningful HTTP responses when invalid input is supplied.

The API also converts the model confidence into the expected 0–1 range for the response schema.

This ensures consistency between the model output and the Pydantic response validation.

#13. Docker Deployment

The FastAPI application was containerized using Docker.

A Dockerfile was created using a Python 3.12 base image.

The container includes:

1.Python environment
2.Required dependencies
3.FastAPI application
4.ResNet18 model
5.Supporting project files

The API runs on port 8000.

The container exposes:
0.0.0.0:8000

The API can be accessed locally using:
http://127.0.0.1:8000

Swagger documentation is available at:
http://127.0.0.1:8000/docs

Docker provides a consistent environment for running and testing the machine learning API.

#14. Load Testing

##14.1 Locust Setup

Locust was used to evaluate the API under concurrent user load.

The load-testing script generated requests for:
1./health
2./predict

The prediction request uploaded a test image to the API.

The test used:

10 concurrent users

The purpose of the test was to measure:

1.Request throughput
2.Response latency
3.Failure rate
4.Endpoint performance
5.API reliability

##14.2 Load Test Results

The successful Locust test produced the following results:

Metric	                              Result
Concurrent Users	                10
Total Requests	                        787
Failures	                         0
Failure Rate	                         0%
Requests per Second	                6.5
Average Response Time	               25.29 ms

Endpoint-level results:

Endpoint	Requests	Failures	Average Response
/health	        402	           0	             8.01 ms
/predict	385	           0	             43.34 ms

Additional /predict latency results:

Metric	                       Result
Median	                        41 ms
95th Percentile	                67 ms
99th Percentile	                96 ms
Maximum	                        141 ms

The load test completed successfully with zero failures.

The results demonstrate that the containerized FastAPI service was able to handle concurrent requests reliably during the test.

#15. Discussion

The project demonstrates the complete lifecycle of an image-classification machine learning system.

The custom CNN provided a useful baseline with 81.62% test accuracy.

The ResNet18 transfer-learning experiment improved performance to 94.36% accuracy and achieved an F1 score of 94.35%.

This improvement demonstrates the effectiveness of using a stronger residual architecture for image classification.

The data-efficiency experiment showed that increasing the training-data size generally improved performance, although it also increased training time.

The model compression experiment introduced 20% sparsity into ResNet18. After fine-tuning, the pruned model achieved 80.49% accuracy and an F1 score of 80.48%.

Although the pruned model experienced a reduction in classification performance compared with the original ResNet18, its measured inference time decreased from 4.2690 seconds to 2.078 seconds.

The pruning experiment therefore highlights the trade-off between predictive performance and computational efficiency.

The deployment stage demonstrated that the model could be exposed through a REST API and packaged inside a Docker container.

Finally, Locust load testing showed zero request failures across 787 requests with 10 concurrent users.

#16. Limitations

Several limitations were identified during the project.

##16.1 Unstructured Pruning

The pruning experiment used unstructured pruning.

Although this creates sparse weights, the number of stored parameters and dense model size did not decrease in the experiment.

Structured pruning or hardware-aware optimization could provide more direct reductions in memory usage and inference cost.

##16.2 Accuracy Reduction After Pruning

The pruned and fine-tuned model achieved lower accuracy than the original ResNet18.

Additional fine-tuning epochs, different learning rates, or a lower pruning ratio could potentially reduce this performance gap.

##16.3 Limited Load-Test Scale

The Locust experiment used 10 concurrent users.

A production system would require testing with substantially higher concurrency and longer test durations.

##16.4 Hardware Dependence

Inference and training times depend on the hardware and execution environment.

Therefore, measured timings should be interpreted as experimental measurements rather than universal performance values.

##16.5 CIFAR-10 Image Resolution

CIFAR-10 contains low-resolution 32 × 32 images.

Consequently, the results may not directly represent performance on high-resolution real-world images.

#17. Future Work

Future improvements could include:

1.Structured pruning
2.Quantization
3.Knowledge distillation
4.More extensive fine-tuning after pruning
5.Hardware-aware optimization
6.GPU-based production inference
7.Batch inference optimization
8.Model versioning
9.Automated CI/CD deployment
10.Authentication for the prediction API
11.Higher-scale load testing
12.Cloud deployment
13.Monitoring and logging
14.Performance profiling
15.Automated model retraining

A future version could also compare lightweight architectures such as MobileNet and EfficientNet with ResNet18.

#18. Conclusion

This project demonstrated an end-to-end machine learning workflow extending from model development to deployment and performance testing.

A custom CNN was first developed as a baseline and achieved 81.62% test accuracy.

ResNet18 transfer learning significantly improved the classification result, achieving 94.36% accuracy and a 94.35% F1 score.

Data-efficiency experiments demonstrated the relationship between training-data volume, model performance, and training time.

A 20% unstructured pruning strategy was then applied to ResNet18. After five epochs of fine-tuning, the pruned model achieved 80.49% accuracy and an F1 score of 80.48%, while its measured 
inference time was reduced to 2.078 seconds.

The model was subsequently integrated into a FastAPI application and containerized using Docker.

Load testing with Locust demonstrated that the deployed API successfully processed 787 requests using 10 concurrent users with zero failures.

Overall, the project demonstrates the practical machine learning lifecycle:

Data
 ↓
Preprocessing
 ↓
Model Development
 ↓
Training
 ↓
Evaluation
 ↓
Optimization
 ↓
Fine-Tuning
 ↓
API Integration
 ↓
Docker Deployment
 ↓
Load Testing

The project therefore provides practical experience across deep learning experimentation, model optimization, API development, containerization, and performance evaluation.

#19. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep Residual Learning for Image Recognition.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

2.Krizhevsky, A. (2009).
Learning Multiple Layers of Features from Tiny Images.
University of Toronto.

3.PyTorch Documentation.
PyTorch: An Open Source Machine Learning Framework.

4.FastAPI Documentation.
FastAPI: Modern, Fast Web Framework for Building APIs with Python.

5.Docker Documentation.
Docker Documentation and Containerization Platform.

6.Locust Documentation.
Locust: An Open Source Load Testing Tool.


