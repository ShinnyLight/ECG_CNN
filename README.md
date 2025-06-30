# 🫀 CNN and Transformer Baselines for PTB-XL ECG Classification

This repository provides PyTorch implementations of two baseline models — a Convolutional Neural Network (CNN) and a Transformer — for multi-label classification of 12-lead ECG signals from the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/). The project includes training, evaluation, and visualization scripts suitable for reproducible experimentation.

---

## 📁 Project Structure

ECG_CNN/
├── src/
│ ├── dataset.py # PTB-XL preprocessing and dataset loader
│ ├── model.py # Model definitions (CNN and Transformer)
├── train_cnn.py # Train CNN baseline
├── train_transformer.py # Train Transformer model
├── evaluate_cnn.py # Evaluate CNN model and generate metrics
├── evaluate_transformer.py # Evaluate Transformer model
├── quick_train_cnn.py # Lightweight CNN debug script
├── plot_class_distribution.py # Visualize dataset label imbalance
├── models/ # Trained model outputs (.pth)
├── results/ # Output CSVs and charts

yaml
Copy
Edit

---

## 🛠️ Environment Setup

We recommend using `conda` (Python 3.10) to manage your environment.

### 1. Create & activate environment

```bash
conda create -n ecgcnn python=3.10
conda activate ecgcnn
2. Install dependencies
bash
Copy
Edit
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other required packages
pip install wfdb pandas scikit-learn tqdm matplotlib
📦 Data Preparation
Download the PTB-XL dataset from PhysioNet:
👉 https://physionet.org/content/ptb-xl/1.0.3/

Extract the data and ensure the following structure:

Copy
Edit
ptbxl_database.csv
scp_statements.csv
records100/
records500/
Update DATA_PATH in your training and evaluation scripts to point to the correct dataset location.

🚀 Model Training
▶️ Train CNN
bash
Copy
Edit
python train_cnn.py
The trained model will be saved to:
models/cnn_model_71.pth

▶️ Train Transformer
bash
Copy
Edit
python train_transformer.py
The trained model will be saved to:
models/transformer_model_71.pth

📊 Evaluation & Visualization
📈 Evaluate CNN
bash
Copy
Edit
python evaluate_cnn.py
Outputs predictions to: results/cnn_predictions.csv

Generates per-class F1-score plot: results/cnn_f1_per_class.png

Prints strict & per-label accuracy, sample-wise precision, recall, and F1

📈 Evaluate Transformer
bash
Copy
Edit
python evaluate_transformer.py
Outputs predictions to: results/transformer_predictions.csv

Generates per-class F1-score plot: results/transformer_f1_per_class.png

Same evaluation metrics as CNN

⚙️ Debug & Analysis Tools
🧪 Quick Debug Script
Use a small subset to quickly verify model functionality and device (GPU):

bash
Copy
Edit
python quick_train_cnn.py
📉 Class Imbalance Visualization
bash
Copy
Edit
python plot_class_distribution.py
This will generate a bar chart of label frequency from the PTB-XL dataset.

📚 Citation
If you use this dataset or code, please cite the original PTB-XL paper:

Wagner, P., Strodthoff, N., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7, 154.
https://doi.org/10.1038/s41597-020-0495-6