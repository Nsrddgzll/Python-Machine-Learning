# Python Machine Learning Code Repository
> Implementation of code examples from "Python Machine Learning" by Sebastian Raschka

## Overview
This repository contains my implementations and notes from working through Sebastian Raschka's "Python Machine Learning" book. The code covers fundamental machine learning concepts, scikit-learn implementations, and deep learning with Python.

## Repository Structure
```
├── chapter01_ml_fundamentals/
│   ├── perceptron_implementation.py
│   ├── adaline_implementation.py
│   └── notes.md
├── chapter02_classification/
│   ├── logistic_regression.py
│   ├── svm_implementation.py
│   └── notes.md
└── ...
```

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- TensorFlow 2.x (for deep learning chapters)

## Installation
1. Clone this repository:
```bash
git clone https://github.com/Nsrddgzll/Python-Machine-Learning
cd python-machine-learning-raschka
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Chapter Progress
- [ ] Chapter 1: Machine Learning - Giving Computers the Ability to Learn from Data
- [ ] Chapter 2: Training Simple ML Algorithms for Classification
- [ ] Chapter 3: A Tour of Machine Learning Classifiers Using Scikit-learn
- [ ] Chapter 4: Building Good Training Datasets - Data Preprocessing
- [ ] Chapter 5: Dimensionality Reduction
- [ ] Chapter 6: Learning Best Practices for Model Evaluation and Hyperparameter Tuning
- [ ] Chapter 7: Combining Different Models for Ensemble Learning
- [ ] Chapter 8: Applying Machine Learning to Sentiment Analysis
- [ ] Chapter 9: Embedding a Machine Learning Model into a Web Application
- [ ] Chapter 10: Predicting Continuous Target Variables with Regression Analysis
- [ ] Chapter 11: Working with Unlabeled Data – Clustering Analysis
- [ ] Chapter 12: Implementing a Multi-layer Artificial Neural Network
- [ ] Chapter 13: Parallelizing Neural Network Training with TensorFlow

## Notes Structure
Each chapter directory contains:
- Implementation of algorithms from scratch (when applicable)
- Scikit-learn implementations
- Comparison studies and visualizations
- Exercise solutions
- Personal notes and insights

## Key Implementations
1. Chapter 1-2:
   - Perceptron implementation from scratch
   - Adaptive Linear Neuron (ADALINE)
   - Logistic Regression implementation

2. Chapter 3-4:
   - Support Vector Machines
   - Decision Trees
   - K-Nearest Neighbors
   - Data preprocessing pipelines

3. Chapter 5-6:
   - Principal Component Analysis (PCA)
   - Linear Discriminant Analysis (LDA)
   - Cross-validation implementations
   - Grid search

4. Later Chapters:
   - Sentiment Analysis
   - Neural Networks from scratch
   - TensorFlow implementations

## Running the Code
1. Navigate to the specific chapter directory
2. Start Jupyter Notebook:
```bash
jupyter notebook
```
3. Open the relevant notebook and run the cells

## Study Notes Template
### Chapter X: [Title]
#### Key Concepts
- Main algorithms covered
- Mathematical foundations
- Practical applications

#### Implementation Details
```python
# Key code snippets and explanations
```

#### Visualization Outputs
- Learning curves
- Decision boundaries
- Model comparisons

#### Personal Notes
- Challenges faced
- Important insights
- Practical tips

## Contributing
This is a personal learning repository, but suggestions and corrections are welcome:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Sebastian Raschka for the excellent book and clear explanations
- The scikit-learn community for their comprehensive documentation
- Python Machine Learning community for additional resources

---
**Note**: This repository contains my personal implementations while studying the book. For official code repositories and materials, please refer to Sebastian Raschka's official GitHub repository.