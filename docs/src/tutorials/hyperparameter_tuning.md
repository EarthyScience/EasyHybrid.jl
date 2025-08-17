```@raw html
---
authors:
  - name: Bernhard Ahrens
    avatar: /assets/Bernhard_Ahrens.png
    link: https://www.bgc-jena.mpg.de/en/bgi/miss
  - name: Lazaro Alonso
    avatar: https://avatars.githubusercontent.com/u/19525261?v=4
    platform: github
    link: https://lazarusa.github.io

---

<Authors />
```

# Hyperparameter tuning
How did we actually get the get started to work?

### 5. Train the Model

```@example quick_start_complete

using CairoMakie

out = train(
    hybrid_model, 
    ds, 
    (); 
    nepochs = 100,               # Number of training epochs
    batchsize = 512,             # Batch size for training
    opt = AdamW(0.001),        # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,           # Scaling for outputs
    patience = 30,               # Early stopping patience
    show_progress=false
)
```

```@raw html
<video src="../training_history_after.mp4" controls="controls" autoplay="autoplay"></video>
```

### 6. Check Results

Evolution of train and validation loss

```@example quick_start_complete
using CairoMakie
EasyHybrid.plot_loss(out, yscale = identity)
```

Check results - what do you think - is it the true Q10 used to generate the synthetic dataset?

```@example quick_start_complete
out.train_diffs.Q10
``` 

Quick scatterplot - dispatches on the output of train

```@example quick_start_complete
EasyHybrid.poplot(out)
```

## Hyperparameter Tuning

EasyHybrid provides built-in hyperparameter tuning capabilities to optimize your model configuration. This is especially useful for finding the best neural network architecture, optimizer settings, and other hyperparameters.

### Basic Hyperparameter Tuning

You can use the `tune` function to automatically search for optimal hyperparameters:

```@example quick_start_complete
using Hyperopt
using Distributed

# Create empty model specification for tuning
mspempty = ModelSpec()

# Define hyperparameter search space
nhyper = 4
ho = @thyperopt for i=nhyper,
    opt = [AdamW(0.01), AdamW(0.1), RMSProp(0.001), RMSProp(0.01)],
    input_batchnorm = [true, false]
    
    hyper_parameters = (;opt, input_batchnorm)
    println("Hyperparameter run: ", i, " of ", nhyper, " with hyperparameters: ", hyper_parameters)
    
    # Run tuning with current hyperparameters
    out = EasyHybrid.tune(
        hybrid_model, 
        ds, 
        mspempty; 
        hyper_parameters..., 
        nepochs = 10, 
        plotting = false, 
        show_progress = false, 
        file_name = "test$i.jld2"
    )
    
    out.best_loss
end

# Get the best hyperparameters
ho.minimizer
printmin(ho)
```

### Manual Hyperparameter Tuning

For more control, you can manually specify hyperparameters and run the tuning:

```@example quick_start_complete
# Run tuning with specific hyperparameters
out_tuned = EasyHybrid.tune(
    hybrid_model, 
    ds, 
    mspempty; 
    opt = RMSProp(eta=0.001), 
    input_batchnorm = true, 
    nepochs = 100,
    monitor_names = [:rb, :Q10],
    hybrid_name="after"
)

# Check the tuned model performance
out_tuned.best_loss
```

```@raw html
<video src="../training_history_after.mp4" controls="controls" autoplay="autoplay"></video>
```

### Key Hyperparameters to Tune

When tuning your hybrid model, consider these important hyperparameters:

- **Optimizer and Learning Rate**: Try different optimizers (AdamW, RMSProp, Adam) with various learning rates
- **Neural Network Architecture**: Experiment with different `hidden_layers` configurations
- **Activation Functions**: Test different activation functions (relu, sigmoid, tanh)
- **Batch Normalization**: Enable/disable `input_batchnorm` and other normalization options
- **Batch Size**: Adjust `batchsize` for optimal training performance
- **Early Stopping**: Set appropriate `patience` values to prevent overfitting

### Best Practices for Hyperparameter Tuning

1. **Start with a small search space** to get a baseline understanding
2. **Use cross-validation** when possible to ensure robust parameter selection
3. **Monitor for overfitting** by tracking validation loss
4. **Save intermediate results** using the `file_name` parameter
5. **Consider computational cost** - more hyperparameters and epochs increase training time

## More Examples

Check out the `projects/` directory for additional examples and use cases. Each project demonstrates different aspects of hybrid modeling with EasyHybrid.