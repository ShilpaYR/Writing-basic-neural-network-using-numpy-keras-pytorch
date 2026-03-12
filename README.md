# Writing Basic Neural Network Using NumPy, Keras, and PyTorch

**Student Name:** Shilpa Yelkur Ramakrishnaiah  
**Student ID:** 019151782  
**Course Assignment:** Write basic neural network using numpy keras pytorch  
**GitHub Repository:** https://github.com/ShilpaYR/Writing-basic-neural-network-using-numpy-keras-pytorch

---

## Assignment Overview

This repository contains all required Google Colab notebooks for the assignment **“Write basic neural network using numpy keras pytorch.”**  
The work covers the same nonlinear regression problem across multiple implementation styles, beginning from a manual low-level approach and progressing toward higher-level framework abstractions.

The assignment requirements include:
- a **NumPy-style manual network** with manual backpropagation and chain rule based gradient propagation,
- a **PyTorch from-scratch version** without using built-in layer classes,
- a **PyTorch class-based built-in version** using modules,
- a **PyTorch Lightning version**,
- and **four TensorFlow variants** covering low-level, built-in layers, Functional API, and model subclassing.

All notebooks use a **3-variable nonlinear synthetic regression problem**, include **training results**, **loss behavior**, **final predictions**, and a **4D-style visualization** using dimensionality reduction for plotting.

---

## Repository Contents

The repository currently contains the following notebooks:  
- `A_numpy_style_manual_backprop (1).ipynb`  
- `B_pytorch_scratch_no_builtin_layers_V2 (1).ipynb`  
- `C_pytorch_class_based_builtin (1).ipynb`  
- `D_pytorch_lightning (1).ipynb`  
- `E1_tensorflow_low_level (1).ipynb`  
- `E2_tensorflow_builtin_layers (1).ipynb`  
- `E3_tensorflow_functional_api (1).ipynb`  
- `E4_tensorflow_high_level_subclassing (1).ipynb`  

---

## Quick Access Table: Colabs and Video Walkthroughs

| Assignment Part | Notebook File | Implementation Type | Video Walkthrough |
|---|---|---|---|
| A | `A_numpy_style_manual_backprop (1).ipynb` | NumPy-style manual neural network with manual backpropagation and `tf.einsum` | https://youtu.be/H7VnsYx73RY |
| B | `B_pytorch_scratch_no_builtin_layers_V2 (1).ipynb` | PyTorch from scratch without built-in layer functionality | https://youtu.be/jx4tdm7OrFI |
| C | `C_pytorch_class_based_builtin (1).ipynb` | PyTorch class-based neural network using built-in modules | https://youtu.be/jx4tdm7OrFI |
| D | `D_pytorch_lightning (1).ipynb` | PyTorch Lightning implementation | https://youtu.be/jx4tdm7OrFI |
| E1 | `E1_tensorflow_low_level (1).ipynb` | TensorFlow low-level implementation without high-level API | https://youtu.be/sS1-PAO4otQ |
| E2 | `E2_tensorflow_builtin_layers (1).ipynb` | TensorFlow built-in layers implementation | https://youtu.be/bSf2RFEcYuU |
| E3 | `E3_tensorflow_functional_api (1).ipynb` | TensorFlow Functional API implementation | https://youtu.be/WupTIjZCzh4 |
| E4 | `E4_tensorflow_high_level_subclassing (1).ipynb` | TensorFlow high-level model subclassing implementation | https://youtu.be/29ZadO7YeRs |

> **Note:** The supplied PyTorch video mapping includes one walkthrough link covering the PyTorch notebook set. If your uploaded YouTube titles are more specific in your final submission, keep the same table format and adjust the links row-by-row.

---

## Notebook-by-Notebook Summary

### A. NumPy-Style Manual Backpropagation
This notebook implements a **3-hidden-layer neural network** for nonlinear regression in a manual style.  
The model uses explicitly defined weights and biases, nonlinear activation functions, manual forward propagation, manual backpropagation, and chain-rule-based gradient computation.  
It also includes:
- synthetic data generation using **3 nonlinear input variables**,
- training and validation loss tracking,
- final regression output evaluation,
- and a **4D-style plot** using PCA for visualization.

### B. PyTorch From Scratch (No Built-In Layers)
This notebook implements the same nonlinear regression task using **PyTorch tensors and autograd**, but **without using built-in layer modules such as `nn.Linear`**.  
The trainable parameters are created manually, the forward pass is explicitly defined, and the parameter updates are controlled directly.

### C. PyTorch Class-Based Built-In Version
This notebook implements the regression network using **PyTorch `nn.Module` and built-in neural network layers**.  
Compared with the scratch implementation, this version is cleaner and more modular, while still keeping the network architecture explicit and interpretable.

### D. PyTorch Lightning Version
This notebook solves the same problem using **PyTorch Lightning**, which reduces training boilerplate and organizes the code into a more structured high-level workflow.  
The implementation demonstrates how the same network can be trained using Lightning modules, validation steps, and trainer-based execution.

### E1. TensorFlow Low-Level Version
This notebook implements the network using **low-level TensorFlow operations** without relying on high-level model-building APIs.  
Weights and biases are manually defined, forward propagation is written explicitly, and training is performed through a custom loop using **`tf.GradientTape()`**.

### E2. TensorFlow Built-In Layers Version
This notebook uses **TensorFlow built-in Dense layers** to construct the same nonlinear regression network.  
It provides a simpler and more readable implementation while keeping the same dataset, architecture, and evaluation flow.

### E3. TensorFlow Functional API Version
This notebook uses the **TensorFlow Functional API** to define the model from input tensors to output tensors.  
This version highlights how neural network graphs can be defined more explicitly while remaining inside TensorFlow’s high-level ecosystem.

### E4. TensorFlow High-Level Subclassing Version
This notebook uses **TensorFlow model subclassing** to define the regression model in an object-oriented way.  
It provides flexibility and clean structure while still using TensorFlow’s higher-level training workflow.

---

## Common Problem Setup Across All Notebooks

All notebooks are based on the same learning objective:
- solve a **nonlinear regression** problem,
- using **3 input variables**,
- with a **3-hidden-layer neural network**,
- and compare how the same problem is implemented across different frameworks and abstraction levels.

### Shared workflow across notebooks
1. Generate synthetic nonlinear regression data using 3 variables.  
2. Split the data into training, validation, and test sets.  
3. Visualize the data using a **4D-style plot**.  
4. Train a 3-hidden-layer neural network.  
5. Track training behavior through loss curves.  
6. Evaluate model performance using final prediction metrics and true-vs-predicted plots.  

### Common evaluation style
The notebooks typically include:
- loss across epochs,
- final train/validation/test behavior,
- regression metrics such as **MSE, RMSE, MAE, and R²**,
- and a final **true vs predicted** scatter plot.

---

## How to Review This Submission

For the Teaching Assistant or Instructor:

1. Open the GitHub repository.  
2. Review the notebook names in the table above.  
3. Use the corresponding YouTube links for each notebook walkthrough.  
4. Open each notebook and verify:
   - execution cells,
   - model implementation,
   - loss curves,
   - output plots,
   - and final regression evaluation.  
5. Refer to the report in the repository for the complete written summary of the assignment work.

---

## Notes on Video Walkthroughs

The video walkthroughs are included to satisfy the assignment requirement of showing:
- the checked-in GitHub repository,
- the executed Google Colab notebooks,
- section-by-section explanation of the code,
- the main logic behind each implementation,
- and the final outputs and results.

These videos are intended to make evaluation faster and clearer by directly mapping each notebook to a corresponding explanation.

---

## Conclusion

This repository demonstrates the implementation of a **basic neural network for nonlinear regression** across multiple frameworks and coding styles:
- manual NumPy-style implementation,
- PyTorch from scratch,
- PyTorch class-based implementation,
- PyTorch Lightning,
- TensorFlow low-level,
- TensorFlow built-in layers,
- TensorFlow Functional API,
- and TensorFlow high-level subclassing.

The goal of the submission is not only to train a working model, but also to compare **manual computation**, **framework-assisted training**, and **high-level abstractions** while keeping the learning problem consistent across all variants.

---

## Direct Links

**GitHub Repository:**  
https://github.com/ShilpaYR/Writing-basic-neural-network-using-numpy-keras-pytorch

**YouTube Walkthroughs:**  
- https://youtu.be/H7VnsYx73RY  
- https://youtu.be/jx4tdm7OrFI  
- https://youtu.be/sS1-PAO4otQ  
- https://youtu.be/bSf2RFEcYuU  
- https://youtu.be/WupTIjZCzh4  
- https://youtu.be/29ZadO7YeRs
