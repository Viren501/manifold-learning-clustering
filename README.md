# High-Dimensional Digit Visualization with t-SNE

An unsupervised machine learning project demonstrating non-linear dimensionality reduction and manifold learning on high-dimensional pixel grids. This workspace scales 64 distinct spatial features down to an expressive 2-dimensional projection using t-SNE, revealing clear geometric clustering profiles among handwritten digit classes.

## Dataset Description
The analysis utilizes the Scikit-learn native digits dataset (`load_digits`), containing **1,797 samples**:
* **Features:** 64 numerical values representing an 8x8 normalized pixel array tracking grayscale values of handwritten symbols.
* **Target Classes:** Categorical mappings matching digits from `0` through `9`.

## Methodology
1. **Data Load & Inspection:** Loaded the raw structure to unpack coordinate boundaries, verifying an array matrix size of `(1797, 64)`.
2. **Feature Scaling:** Implemented `StandardScaler` to bring feature variances to uniform ranges, optimizing step distances during embedding generation.
3. **Manifold Embedding Extraction:** Initialized Scikit-learn's non-linear `TSNE` operator, setting parameters to `n_components=2` and anchoring reproducibility with a fixed random seed. 
4. **Data Transformation:** Fit and transformed the scaled structure to compress structural vectors into a 2D matrix shape of `(1797, 2)`.
5. **Scatter Visualization:** Constructed an expressive structural plot utilizing unique color classifications for classes 0–9 to visualize separation qualities.

## Results & Visualization Analysis
* **Dimensional Compression:** Successfully minimized pixel representation arrays from a massive 64-dimensional space into 2 intuitive components without sacrificing neighborhood patterns.
* **Clustering Qualities:** The outputted Matplotlib projection demonstrates remarkable group separation. Rather than mixing loosely like typical linear reductions (e.g., PCA), the non-linear t-SNE model cleanly isolated spatial dependencies, mapping digit tokens from classes 0 through 9 into tightly bounded, distinct vector clusters on the visualization canvas.

## How to Run Locally
1. Clone this repository to your local computer environment workspace.
2. Install standard scientific dependencies via pip:
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
