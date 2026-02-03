# Tutorials Overview

Welcome to the EasyHybrid.jl tutorials! This section provides comprehensive guides and examples to help you get started with hybrid modeling—combining process-based models with neural networks.

## Getting Started

If you're new to EasyHybrid.jl, we recommend starting with the [Getting Started](../get_started.md) guide, which walks through building your first hybrid model step by step.

## Tutorial Topics

### Core Tutorials

#### [Exponential Respiration Model](exponential_res.md)
Learn to build a hybrid model for soil respiration using an exponential temperature relationship. This tutorial demonstrates:
- Creating synthetic data with exponential temperature response
- Defining a process-based model (`Expo_resp_model`)
- Configuring model parameters and constructing hybrid models
- Training and evaluating your model

**Best for**: Understanding the fundamentals of hybrid modeling with a simple, well-documented example.

#### [LSTM Hybrid Model](example_synthetic_lstm.md)
Explore advanced neural network architectures by building a hybrid model with LSTM (Long Short-Term Memory) networks. This tutorial covers:
- Using LSTM networks for sequence modeling
- Configuring feedforward vs. recurrent architectures
- Working with sequence data (input/output windows, lead times)
- Comparing LSTM and standard neural network performance

**Best for**: Working with time series data that requires memory of past states.

### Model Evaluation and Optimization

#### [Cross-Validation](folds.md)
Implement k-fold cross-validation to robustly evaluate your hybrid models. Learn how to:
- Create and manage data folds
- Train models across multiple validation splits
- Parallelize cross-validation training
- Organize results from multiple model runs

**Best for**: Ensuring your model generalizes well and avoiding overfitting.

#### [Hyperparameter Tuning](hyperparameter_tuning.md)
Optimize your model's performance using Hyperopt.jl for automated hyperparameter search. This tutorial shows:
- Setting up hyperparameter search spaces
- Using the `tune` function for optimization
- Comparing model performance before and after tuning
- Best practices for hyperparameter selection

**Best for**: Finding optimal model configurations and improving performance.

### Advanced Topics

#### [Losses and LoggingLoss](losses.md)
Deep dive into the loss function system in EasyHybrid.jl. Learn about:
- Predefined loss functions (MSE, MAE, NSE)
- Creating custom loss functions
- Handling missing values and uncertainty
- Using `LoggingLoss` for training and evaluation
- Passing additional arguments and keyword arguments to losses

**Best for**: Customizing model training and implementing domain-specific loss functions.

#### [Slurm Jobs](slurm.md)
Run EasyHybrid.jl models on HPC clusters using Slurm job scheduling. This tutorial provides:
- Example Slurm batch scripts
- Configuring Julia for cluster environments
- Running array jobs for parallel experiments
- Resource allocation best practices

**Best for**: Scaling up training to high-performance computing environments.

## Choosing the Right Tutorial

- **New to EasyHybrid?** → Start with [Exponential Respiration Model](exponential_res.md)
- **Working with time series?** → Check out [LSTM Hybrid Model](example_synthetic_lstm.md)
- **Need robust evaluation?** → See [Cross-Validation](folds.md)
- **Want to optimize performance?** → Try [Hyperparameter Tuning](hyperparameter_tuning.md)
- **Customizing training?** → Read [Losses and LoggingLoss](losses.md)
- **Running on clusters?** → Follow [Slurm Jobs](slurm.md)

## Next Steps

After completing the tutorials, explore the [Research](../research/overview.md) section to see real-world applications, or dive into the [API Reference](../api.md) for detailed function documentation.
