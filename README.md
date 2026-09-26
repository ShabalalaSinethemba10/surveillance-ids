## Overview

This project implements a **Stacked Ensemble** intrusion detection system for IP cameras and IoT devices. It uses the **N-BaIoT dataset** containing real traffic from 9 commercial IoT devices infected with the Mirai and BASHLITE botnets.

We also trained two hybrid deep learning baselines (CNN-BiLSTM and CNN-GRU) for comparison. However, PyTorch's ONNX export has known limitations with recurrent layers, so the **Stacked Ensemble is the deployed model**.

### Key Features

- **Stacked Ensemble**: Random Forest + XGBoost + SVM + Logistic Regression meta-learner
- **Browser Deployment**: ONNX Runtime Web for real-time client-side inference
- **Edge-Ready**: Model is 0.55 MB and runs in <2 ms per sample
- **Production-Ready**: 99.98% accuracy on 41,434 test samples