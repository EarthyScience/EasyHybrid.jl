import{_ as s,c as n,o as e,aA as p}from"./chunks/framework.D_UQp95l.js";const u=JSON.parse('{"title":"Generic Hybrid","description":"","frontmatter":{},"headers":[],"relativePath":"generic_hybrid.md","filePath":"generic_hybrid.md","lastUpdated":null}'),i={name:"generic_hybrid.md"};function r(t,a,l,o,c,d){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="Generic-Hybrid" tabindex="-1">Generic Hybrid <a class="header-anchor" href="#Generic-Hybrid" aria-label="Permalink to &quot;Generic Hybrid {#Generic-Hybrid}&quot;">​</a></h1><p>This page demonstrates how to use EasyHybrid to create a hybrid model for ecosystem respiration. This example shows the key concepts of EasyHybrid:</p><ol><li><p><strong>Process-based Model</strong>: The <code>RbQ10</code> function represents a classical Q10 model for respiration with base respiration <code>rb</code> and <code>Q10</code> which describes the factor by respiration is increased for a 10 K change in temperature</p></li><li><p><strong>Neural Network</strong>: Learns to predict the basal respiration parameter <code>rb</code> from environmental conditions</p></li><li><p><strong>Hybrid Integration</strong>: Combines the neural network predictions with the process-based model to produce final outputs</p></li><li><p><strong>Parameter Learning</strong>: Some parameters (like <code>Q10</code>) can be learned globally, while others (like <code>rb</code>) are predicted per sample</p></li></ol><p>The framework automatically handles the integration between neural networks and mechanistic models, making it easy to leverage both data-driven learning and domain knowledge.</p><h2 id="Quick-Start-Example" tabindex="-1">Quick Start Example <a class="header-anchor" href="#Quick-Start-Example" aria-label="Permalink to &quot;Quick Start Example {#Quick-Start-Example}&quot;">​</a></h2><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>using EasyHybrid</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 1. Setup and Data Loading</span></span>
<span class="line"><span># Load synthetic dataset</span></span>
<span class="line"><span>ds = load_timeseries_netcdf(&quot;https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc&quot;)</span></span>
<span class="line"><span>ds = ds[1:20000, :]  # Use subset for faster execution</span></span>
<span class="line"><span>first(ds, 5)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 2. Define the Process-based Model</span></span>
<span class="line"><span># RbQ10 model: Respiration model with Q10 temperature sensitivity</span></span>
<span class="line"><span>RbQ10 = function(;ta, Q10, rb, tref = 15.0f0)</span></span>
<span class="line"><span>    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))</span></span>
<span class="line"><span>    return (; reco, Q10, rb)</span></span>
<span class="line"><span>end</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 3. Configure Model Parameters</span></span>
<span class="line"><span># Parameter specification: (default, lower_bound, upper_bound)</span></span>
<span class="line"><span>parameters = (</span></span>
<span class="line"><span>    rb  = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]</span></span>
<span class="line"><span>    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity - describes factor by which respiration is increased for 10 K increase in temperature [-]</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 4. Construct the Hybrid Model</span></span>
<span class="line"><span># Define input variables</span></span>
<span class="line"><span>forcing = [:ta]                    # Forcing variables (temperature)</span></span>
<span class="line"><span>predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation)</span></span>
<span class="line"><span>target = [:reco]                   # Target variable (respiration)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Parameter classification</span></span>
<span class="line"><span>global_param_names = [:Q10]        # Global parameters (same for all samples)</span></span>
<span class="line"><span>neural_param_names = [:rb]         # Neural network predicted parameters</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Construct hybrid model</span></span>
<span class="line"><span>hybrid_model = constructHybridModel(</span></span>
<span class="line"><span>    predictors,              # Input features</span></span>
<span class="line"><span>    forcing,                 # Forcing variables</span></span>
<span class="line"><span>    target,                  # Target variables</span></span>
<span class="line"><span>    RbQ10,                  # Process-based model function</span></span>
<span class="line"><span>    parameters,              # Parameter definitions</span></span>
<span class="line"><span>    neural_param_names,      # NN-predicted parameters</span></span>
<span class="line"><span>    global_param_names,      # Global parameters</span></span>
<span class="line"><span>    hidden_layers = [16, 16], # Neural network architecture</span></span>
<span class="line"><span>    activation = swish,      # Activation function</span></span>
<span class="line"><span>    scale_nn_outputs = true, # Scale neural network outputs</span></span>
<span class="line"><span>    input_batchnorm = true   # Apply batch normalization to inputs</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 5. Train the Model</span></span>
<span class="line"><span># using WGLMakie # to see an interactive and automatically updated train_board figure</span></span>
<span class="line"><span>out = train(</span></span>
<span class="line"><span>    hybrid_model, </span></span>
<span class="line"><span>    ds, </span></span>
<span class="line"><span>    (); </span></span>
<span class="line"><span>    nepochs = 100,           # Number of training epochs</span></span>
<span class="line"><span>    batchsize = 512,         # Batch size for training</span></span>
<span class="line"><span>    opt = RMSProp(0.001),   # Optimizer and learning rate</span></span>
<span class="line"><span>    monitor_names = [:rb, :Q10], # Parameters to monitor during training</span></span>
<span class="line"><span>    yscale = identity,       # Scaling for outputs</span></span>
<span class="line"><span>    patience = 30            # Early stopping patience</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Check results</span></span>
<span class="line"><span>out.train_diffs.Q10</span></span>
<span class="line"><span></span></span>
<span class="line"><span>using CairoMakie</span></span>
<span class="line"><span>EasyHybrid.poplot(out)</span></span>
<span class="line"><span>EasyHybrid.plot_loss(out)</span></span>
<span class="line"><span>EasyHybrid.plot_parameters(out)</span></span></code></pre></div><h2 id="More-Examples" tabindex="-1">More Examples <a class="header-anchor" href="#More-Examples" aria-label="Permalink to &quot;More Examples {#More-Examples}&quot;">​</a></h2><p>Check out the <code>projects/</code> directory for additional examples and use cases. Each project demonstrates different aspects of hybrid modeling with EasyHybrid.</p>`,8)]))}const b=s(i,[["render",r]]);export{u as __pageData,b as default};
