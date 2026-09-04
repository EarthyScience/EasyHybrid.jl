# EasyHybrid.jl Interactive Tutorial Hour

A monthly, interactive tutorial series on hybrid modeling with `EasyHybrid.jl`.

The goal is not only to learn the EasyHybrid API, but to understand how scientific assumptions can be translated into hybrid models.

Each session should combine a short conceptual introduction, walk through, and hands-on discussion. Whenever possible, sessions should use or extend one of the existing EasyHybrid tutorials.

---

## General Format

Each session is approximately 45 minutes.

A suggested structure is:

* **10 min:** Concept and introduction
* **15–20 min:** Walk through an EasyHybrid tutorial
* **20–25 min:** Interactive modification, implementation, or participant model
* **5–10 min:** Discussion, questions, and ideas for future sessions

A central part of the series is **Bring Your Own Model**:

One participant introduces a scientific model, equation, or modeling idea. The group discusses how it could be represented in EasyHybrid and tries to develop or implement a first version together.

The focus should be on questions such as:

* What is known mechanistically?
* What should be learned from data?
* Which variables are forcings?
* Which variables should be predictors?
* Which parameters should be global?
* Which parameters could vary with environmental conditions?
* Where could a neural network replace or extend an uncertain process representation?

---

# Syllabus

## 1. From a Mechanistic Model to a Hybrid Model

**Tutorial:** Exponential Respiration Model (`exponential_res.md`)

### Main question

How do we turn an ordinary process-based model into a hybrid model?

### Example

Start from the simple soil respiration model

```math
Resp = Resp0 \times \exp(kT)
```

with

```math
Resp0 = f(SM)
```

where:

* `T` is temperature
* `SM` is soil moisture
* `Resp0` is basal respiration
* `k` controls temperature sensitivity

### Step 1: Define the mechanistic model

```julia
function Expo_resp_model(; T, Resp0, k)
    Resp_obs = Resp0 .* exp.(k .* T)
    return (; Resp_obs, Resp0, k)
end
```

Discuss what the model assumes and what each parameter means.

### Step 2: Run it as a process model

Before introducing any neural network, use the function directly.

Discuss:

* Which quantities are measured?
* Which quantities are parameters?
* Which parts of the model do we trust?
* Which relationships are uncertain?

### Step 3: Make one component hybrid

Assume that temperature dependence is known mechanistically, but the environmental controls on basal respiration are not.

Let

```math
Resp0 = NN(SM)
```

while keeping

```math
Resp = Resp0 \times \exp(kT)
```

The resulting model is:

```text
SM ──► Neural network ──► Resp0
                         │
T ──────────────────────►│
                         ▼
                Resp0 × exp(k × T)
                         │
                         ▼
                       Resp
```

### Step 4: Translate the idea into EasyHybrid

```julia
targets = [:Resp_obs]

forcings = [:T]

predictors = (
    Resp0 = [:SM],
)

global_param_names = [:k]
```

Discuss why:

* `T` is a forcing
* `SM` is a predictor
* `Resp0` is predicted by the neural network
* `k` is globally learned
* `Resp_obs` is the target

Then construct the model:

```julia
hybrid_model = constructHybridModel(
    predictors,
    forcings,
    targets,
    Expo_resp_model,
    parameters,
    global_param_names,
    hidden_layers = [16, 16],
    activation = sigmoid,
)
```

### Interactive exercise

Change the scientific assumptions.

For example:

1. `Resp0` neural, `k` global
2. `Resp0` and `k` global
3. `Resp0` global, `k` neural
4. `Resp0` depends on several environmental variables
5. both parameters vary in space or time

For every version, ask:

> What scientific hypothesis does this model represent?

Training can be demonstrated briefly, but optimizer choices, losses, batch sizes, and tuning are intentionally left for later sessions.

---

## 2. Bring Your Own Model: Turning a Scientific Idea into EasyHybrid

One participant brings a simple scientific model, equation, or modeling idea.

Examples might include:

* decomposition
* soil respiration
* photosynthesis
* plant growth
* microbial processes
* carbon allocation
* hydrology
* nutrient cycling

### Format

The participant has around 5–10 minutes to explain:

1. What quantity should be predicted?
2. What process is being represented?
3. What is already known mechanistically?
4. Which process relationship is uncertain?
5. What data are available?

The group then develops the hybrid formulation together.

### Questions for the group

Identify:

```text
Targets
Forcings
Predictors
Global parameters
Neural parameters
Fixed parameters
```

Then try to write:

```julia
function process_model(; ...)
    ...
end
```

and construct an initial EasyHybrid model.

The goal is not necessarily to finish a working model within one hour.

The goal is to learn how to translate a scientific idea into a hybrid model architecture.

---

## 3. Parameters: Fixed, Global, and Neural

### Main question

What should the neural network actually learn?

Use the respiration example and one of the participant models from Session 2.

Compare:

```text
fixed parameter
global learned parameter
neural parameter
```

Discuss what each choice implies scientifically.

Topics:

* parameter bounds
* identifiability
* environmental variability
* global versus sample-specific parameters
* multiple neural parameters
* different predictors for different parameters

### Interactive component

Take one participant model and construct several competing hybrid hypotheses.

For example:

```text
Model A:
k = global

Model B:
k = NN(SM)

Model C:
k = NN(SM, T)

Model D:
k = global
Resp0 = NN(SM)
```

Discuss which hypotheses are scientifically plausible.

---

## 4. Data Flow: Predictors, Forcings, Targets, and Outputs

### Main question

How does information move through a hybrid model?

Topics:

* predictors
* forcings
* targets
* auxiliary variables
* parameter-specific predictor sets
* multiple outputs
* avoiding information leakage

Draw the computational graph of a model before implementing it.

### Bring Your Own Model

Use a participant model with several environmental variables and jointly decide which variables should enter where.

---

## 5. Training a Hybrid Model

### Main question

What exactly are we optimizing?

Topics:

* differentiating through the process model
* neural network parameters
* global model parameters
* optimization
* batches
* training and validation data
* early stopping

Start from a previously developed model rather than introducing a new example.

The emphasis should remain on understanding the optimization problem rather than optimizer details.

---

## 6. Loss Functions and Likelihoods

**Tutorial:** `losses.md`

### Main question

What does it mean for a hybrid model to fit the data well?

Topics:

* MSE
* MAE
* NSE
* custom loss functions
* likelihood-based losses
* weighting observations
* missing values
* multiple targets

Discuss how the choice of loss function changes what the model learns.

### Interactive component

Fit the same hybrid model using several losses and compare the results.

---

## 7. Evaluating Hybrid Models

### Main question

Did the hybrid model actually learn something useful?

Topics:

* training versus validation performance
* residuals
* predicted parameters
* parameter distributions
* process interpretation
* extrapolation
* comparing hybrid and mechanistic baselines

Where possible, evaluate both:

```text
prediction quality
```

and

```text
scientific plausibility of learned parameters
```

---

## 8. Bring Your Own Model: More Complex Hybrid Models

Return to participant models, now allowing more complex structures.

Possible topics:

* several neural parameters
* several outputs
* interactions between processes
* intermediate model variables
* hierarchical model structures
* coupling several mechanistic components

The group selects one idea and tries to implement it together.

---

## 9. Cross-Validation and Generalization

**Tutorial:** Cross-validation tutorial

### Main question

Does the hybrid model work outside the data it was trained on?

Topics:

* random folds
* site-level splits
* temporal splits
* spatial extrapolation
* environmental extrapolation
* avoiding leakage

Discuss why ordinary random train/test splitting may be insufficient for environmental models.

---

## 10. Neural Network Architectures

### Main question

When is a simple feed-forward network not enough?

Topics:

* MLPs
* LSTMs
* sequence models
* Transformers
* memory and state
* temporal predictors

**Tutorial:** Sequence Hybrid Models

Start from the same hybrid modeling principles developed in the earlier sessions.

The neural architecture changes, but the process model remains part of the computational graph.

---

## 11. Hyperparameter Tuning and Computational Experiments

**Tutorial:** `hyperparameter_tuning.md`

Topics:

* hidden layers
* learning rates
* batch size
* architecture
* model comparison
* reproducibility
* automated tuning

Discuss the difference between:

```text
scientific model choices
```

and

```text
machine-learning hyperparameters
```

The former should generally not be selected solely by prediction performance.

---

## 12. Scaling Up: GPU, HPC, and Larger Experiments

**Tutorials:**

* `gpu.md`
* `slurm.md`

Topics:

* GPUs
* parallel training
* HPC
* SLURM
* repeated model runs
* ensembles
* large datasets

This session can use one of the models developed during the previous sessions and scale it from an interactive example to a larger experiment.

---

# Recurring Bring Your Own Model Slot

Rather than waiting until the end of the series, participant models should become a recurring element.

A useful rhythm could be:

```text
Session 1
Core EasyHybrid example

Session 2
Bring Your Own Model

Session 3
Core concept + participant model

Session 4
Core concept + participant model

Session 5
Bring Your Own Model

...
```

Participants can volunteer beforehand with a very small model idea.

The model does not need to be fully developed. Even something as simple as

```math
R = kC
```

or

```math
GPP = f(light, temperature)
```

is enough.

The interesting part is the discussion:

> Which part should remain mechanistic, and which part should be learned?

---

# Guiding Principle

The tutorial series should teach hybrid modeling through **scientific modeling decisions rather than EasyHybrid API features**.

Instead of asking:

> How do I use `constructHybridModel`?

ask:

> Which environmental dependency do I not know well enough to specify mechanistically?

Instead of asking:

> Which variables are predictors?

ask:

> Which information should the neural network be allowed to use to infer this process parameter?

Instead of asking:

> Should this parameter be neural or global?

ask:

> Do we expect this parameter to remain constant, or should it vary systematically with environmental conditions?

EasyHybrid then becomes the tool for expressing those scientific hypotheses.
