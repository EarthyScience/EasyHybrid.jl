# Transformer Forecasting Approaches

When applying Transformers to time-series or weather data, especially when predicting targets (like Net Ecosystem Exchange) from covariates (like Temperature, Precipitation), the architecture of the sequence generation loop is critical.

Currently, `EasyHybrid` uses **Approach 1: Direct Multi-Step Forecasting**. 

## 1. Direct Multi-Step Forecasting (Currently Used)
Instead of predicting step $t+1$, feeding it back, and predicting $t+2$, the model takes a historical window (e.g., the last 7 days of covariates) and **predicts the entire forecast horizon in a single forward pass**.

**Why we use it:**
*   **No Error Accumulation**: A poor prediction at step 1 does not cascade into step 2.
*   **High Performance**: Requires only a single forward pass, completely eliminating the slow token-by-token loop.
*   **Simplicity**: No `KVCache` or complex autoregressive state tracking is needed.

This is the state-of-the-art approach for most regression/forecasting tasks where a physics-simulator style simulation is not strictly required. Models like Informer and Autoformer utilize this approach.

---

## Future Directions (Autoregressive Architectures)

If the project scope expands to full physical weather simulation, the following approaches can be implemented. The core infrastructure for this (`KVCache` and step-by-step attention) is already built into `EasyHybrid` but currently archived (commented out) in the codebase.

### 2. Full State Autoregressive Forecasting (The Simulator Approach)
To step forward through time infinitely, the model must predict **everything**. To predict the target at step $t+2$, it needs the covariates at step $t+1$. Since those aren't available, the model is trained to predict both the target *and* the covariates simultaneously. The entire output state is then fed back as the input for the next step.

*   **Examples**: Pangu-Weather, FourCastNet.
*   **Pros**: Allows forecasting indefinitely into the future, simulating underlying physics step-by-step.
*   **Cons**: Error accumulation is severe; requires a highly stable model.

### 3. Known Future Covariates (The Hybrid Approach)
If some covariates are known deterministically in the future (like Time of Day, Solar Radiation) or are provided by an external weather forecast, the model can operate autoregressively on the target alone. 
It takes the known future covariates for step $t+1$, combines them with its own predicted target from step $t$, and feeds that combined vector into the next step.

*   **Pros**: Leverages known deterministic variables to ground the model, reducing error accumulation compared to Approach 2.
*   **Cons**: Requires a carefully designed feedback loop to merge predictions with known covariates at every step.
