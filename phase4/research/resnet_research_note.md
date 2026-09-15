# Research Note — Deep Residual Learning for Image Recognition

## 1. Research Paper

**Title:** Deep Residual Learning for Image Recognition

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

**Publication:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

**Paper:** https://arxiv.org/abs/1512.03385

---

## 2. Abstract / Summary

Deep neural networks have achieved strong performance in image recognition, but increasing the depth of a conventional network can make the network increasingly difficult to optimize. The authors proposed a residual learning framework to address this problem.

Instead of asking a group of layers to directly learn an underlying mapping, the residual network learns a residual function with respect to the input. This is implemented using shortcut or skip connections that allow the input to bypass one or more layers.

The proposed Residual Network, commonly known as ResNet, made it possible to successfully train substantially deeper neural networks while maintaining strong accuracy. The paper demonstrated ResNet models with depths up to 152 layers on ImageNet and also investigated very deep networks on CIFAR-10.

The research showed that residual networks are easier to optimize and can achieve improved recognition accuracy as network depth increases.

---

## 3. Problem Addressed

A major problem addressed by the paper is the difficulty of training very deep neural networks.

Simply increasing the number of layers does not always improve performance. Very deep conventional networks can suffer from optimization difficulties and degradation problems, where adding more layers can result in higher training error.

The authors investigated a new architecture that would allow deeper networks to be trained effectively.

---

## 4. Main Idea

The main idea of ResNet is residual learning.

Instead of directly learning a desired mapping:

    H(x)

the network learns a residual function:

    F(x) = H(x) - x

The desired mapping can then be represented as:

    H(x) = F(x) + x

The input x is passed through a shortcut connection and added to the output of the residual layers.

This creates the basic residual block:

    Input x
       |
       +----------------------+
       |                      |
       v                      |
    Conv Layer                |
       |                      |
       v                      |
    Conv Layer                |
       |                      |
       +-----------> Add <----+
                       |
                       v
                    Output

The shortcut connection provides a direct path for information and gradients through the network.

---

## 5. Residual Learning

The residual learning framework is the central contribution of the paper.

A conventional stack of layers attempts to learn the complete transformation of the input. In a residual block, the layers instead learn the residual component while the original input is passed through a shortcut connection.

Mathematically:

    y = F(x, {Wi}) + x

where:

- x is the input
- F(x, {Wi}) represents the residual mapping
- Wi represents the learned weights
- y is the output of the residual block

If the input and output dimensions are different, a projection shortcut can be used to match their dimensions.

---

## 6. Why Skip Connections Are Important

Skip connections help information and gradients travel through the network more directly.

This is particularly useful when networks become very deep.

The shortcut connection allows the network to preserve useful information from earlier layers while the residual layers learn additional transformations.

This helps reduce the optimization difficulty associated with very deep conventional networks.

---

## 7. ResNet Architecture

The paper introduced residual building blocks that can be stacked to construct deep neural networks.

The architecture generally consists of:

1. Initial convolutional layer
2. Multiple residual blocks
3. Downsampling stages
4. Global average pooling
5. Fully connected classification layer

Different ResNet variants are created by changing the number of residual blocks and therefore the total network depth.

Examples include:

- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

ResNet-18 is a relatively lightweight version compared with the deeper variants and is suitable for image-classification tasks where computational efficiency is important.

---

## 8. ResNet and CIFAR-10

The original research also evaluated residual networks on CIFAR-10.

CIFAR-10 contains 60,000 color images belonging to 10 classes. The dataset contains 50,000 training images and 10,000 test images.

The classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The paper investigated residual networks with different depths on CIFAR-10 and showed that very deep residual networks could be effectively optimized.

This makes the research paper directly relevant to our CIFAR-10 image-classification project.

---

## 9. Relevance to Our Project

Our project uses a ResNet18-based image classification model for the CIFAR-10 dataset.

The research paper provides the theoretical foundation for the architecture used in our implementation.

The relationship between the research paper and our project can be summarized as follows:

Research Paper:
Deep Residual Learning

        ↓

Residual / Skip Connections

        ↓

ResNet Architecture

        ↓

ResNet18

        ↓

CIFAR-10 Image Classification

        ↓

Model Evaluation

        ↓

Model Compression / Pruning

        ↓

FastAPI Deployment

The original paper focuses on residual learning and deep image recognition, while our project applies a ResNet18 architecture to CIFAR-10 and extends the work with model evaluation, compression/pruning, and API deployment.

---

## 10. Transfer Learning Connection

Our implementation uses a ResNet18 model as the main architecture for image classification.

Transfer learning allows knowledge learned from a large image dataset to be reused for another image-classification task.

Instead of learning every visual feature from the beginning, a pretrained model can provide useful low-level and mid-level visual representations.

The model can then be adapted to the target classification task.

For CIFAR-10, the final classification layer is adapted to predict 10 classes.

This approach reduces the amount of training required compared with training a large network completely from scratch.

---

## 11. Model Compression Connection

Another part of our project is model compression through pruning.

The goal of pruning is to reduce unnecessary model parameters while attempting to maintain acceptable predictive performance.

The research paper provides the architectural foundation for our ResNet18 model, while our project investigates how the model can subsequently be optimized for deployment.

The overall workflow is therefore:

    ResNet18
       ↓
    Train / Fine-tune
       ↓
    Evaluate
       ↓
    Pruning
       ↓
    Evaluate compressed model
       ↓
    Compare performance
       ↓
    Deploy through API

---

## 12. Experimental Comparison

The project evaluates the original model and the optimized/pruned model.

The following metrics are considered:

- Accuracy
- F1 Score
- Parameter Count
- Model Size
- Inference Time
- Training Time

The exact experimental values are reported in the technical report.
| Metric | Original ResNet18 | Pruned + Fine-Tuned ResNet18 |

| Accuracy | 94.36% | 80.49% |
| F1 Score | 94.35% | 80.48% |
| Parameters | 11,181,642 | 11,181,642 |
| Model Size | 42.69 MB | 42.69 MB |
| Inference Time | 4.2690 s | 2.078 s |
| Training/Fine-Tuning Time | 33.23 min | 1.96 min |
| Sparsity | 0% | 20% |


## Load Testing Results

The deployed FastAPI service was tested using Locust with 10 concurrent users. A total of 787 requests were completed with zero failures, resulting in a 0% failure rate.

The overall request rate was 6.5 requests per second with an average response time of 25.29 ms.

The `/health` endpoint handled 402 requests with an average response time of 8.01 ms. The `/predict` endpoint handled 385 requests with an average response time of 43.34 ms. The prediction endpoint recorded a 95th percentile latency of 67 ms and a maximum latency of 141 ms.

| Endpoint | Requests | Failures | Avg. Response |
| `/health` | 402 | 0 | 8.01 ms |
| `/predict` | 385 | 0 | 43.34 ms |
| **Total** | **787** | **0** | **25.29 ms** |

## 13. Key Findings from the Research Paper

The main findings of the original research are:

1. Very deep conventional networks can become difficult to optimize.

2. Residual learning provides a practical solution to this optimization problem.

3. Shortcut connections allow information to pass through the network more directly.

4. Residual networks can be successfully trained at significantly greater depths.

5. ResNet achieved strong results on major image-recognition benchmarks.

6. The approach was also successfully evaluated on CIFAR-10.

7. Increasing network depth with residual learning can provide improved recognition performance.

---

## 14. Key Findings for Our Project

The research provides the theoretical basis for our use of ResNet18.

Our project extends the basic image-classification architecture by focusing not only on classification performance but also on practical deployment.

The project therefore considers:

- Classification accuracy
- F1 score
- Model complexity
- Model size
- Inference performance
- Model compression
- API deployment
- Load testing

This makes the project a practical application and extension of the residual-learning architecture.

---

## 15. Limitations

The original research primarily focuses on the residual learning architecture and its performance on image-recognition benchmarks.

Our project has a different objective because it also considers deployment and model efficiency.

The project therefore needs to balance:

- Accuracy
- Model size
- Computational requirements
- Inference speed
- Deployment requirements

Model compression can potentially reduce computational requirements, but aggressive compression may also affect predictive performance.

---

## 16. Conclusion

The paper "Deep Residual Learning for Image Recognition" introduced a residual learning framework that made it easier to train substantially deeper neural networks.

The key innovation is the use of shortcut connections that allow layers to learn residual mappings instead of directly learning the complete transformation.

This research forms the foundation of the ResNet family of architectures.

Our project applies the ResNet18 architecture to CIFAR-10 image classification and further investigates model evaluation, pruning/compression, and API deployment.

Therefore, the original ResNet research provides the theoretical foundation, while our project demonstrates its practical implementation and deployment-oriented optimization.

---

## 17. Reference

He, K., Zhang, X., Ren, S., & Sun, J. (2016).

"Deep Residual Learning for Image Recognition."

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

arXiv:1512.03385.

https://arxiv.org/abs/1512.03385