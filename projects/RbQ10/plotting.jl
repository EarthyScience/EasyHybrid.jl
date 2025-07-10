using AlgebraOfGraphics

"""
    plot_scatter(out, what = "validation")

Create scatter plots comparing observed vs predicted values for model outputs.

# Arguments
- `out`: Training output object containing `val_obs_pred` or `train_obs_pred` DataFrames
- `what`: String indicating which data to plot ("validation" or "train")

# Returns
- A Makie figure with scatter plots showing observed vs predicted values for each output variable

# Details
The function automatically:
- Computes R² values for each variable
- Creates a faceted plot layout
- Adds R² annotations to each subplot
"""
function plot_scatter(out, what = "validation")
    # ─────────────────────────────────────────────────────────────
    # 1. your existing vars + data
    # ─────────────────────────────────────────────────────────────
    if what == "validation"
        df = out.val_obs_pred
    elseif what == "train"
        df = out.train_obs_pred
    else
        error("'what' must be either 'validation' or 'train'")
    end

    # Get column names and find index of "index" column
    cols = names(df)
    idx = findfirst(x -> x == "index", cols)
    
    if idx === nothing
        error("DataFrame must contain an 'index' column to separate observed and predicted values")
    end
    
    # Split into y vars (before index) and x vars (after index)
    yvars = Symbol.(cols[1:idx-1])
    xvars = Symbol.(cols[idx+1:end])

    # ─────────────────────────────────────────────────────────────
    # 2. compute R² per variable, dropping NaN/missing first
    # ─────────────────────────────────────────────────────────────
    NSE_list     = Float64[]
    xmin_list   = Float64[]
    xrange_list = Float64[]
    ymax_list   = Float64[]
    yrange_list = Float64[]

    for (y, x) in zip(yvars, xvars)

        yobs = df[!, y]
        ypred = df[!, x]

        # 2a) build mask: drop missing or NaN in either vector
        good = .!ismissing.(yobs) .& .!ismissing.(ypred) .&
               .!isnan.(yobs)    .& .!isnan.(ypred)

        yobs_clean = yobs[good]
        ypred_clean = ypred[good]

        yall = vcat(yobs_clean, ypred_clean)

        # 2b) compute NSE = 1 - SS_res/SS_tot
        ss_res = sum((yobs_clean .- ypred_clean).^2)
        ss_tot = sum((yobs_clean .- mean(yobs_clean)).^2)
        push!(NSE_list, 1 - ss_res/ss_tot)

        # 2c) for positioning the annotation
        push!(xmin_list, minimum(ypred_clean))
        push!(xrange_list, maximum(ypred_clean) - minimum(ypred_clean))
        push!(ymax_list, maximum(yall))
        push!(yrange_list, maximum(yobs_clean) - minimum(yobs_clean))
    end

    # 2d) make string labels
    NSE_labels = ["NSE=$(round(r, digits=2))" for r in NSE_list]

    NSE_df = DataFrame(
      dims1    = string.(yvars[1:length(NSE_list)]),   # must match your facet key
      NSE_label = NSE_labels,
      x        = xmin_list .+ 0.05 .* xrange_list,
      y        = ymax_list .- 0.05 .* yrange_list
    )

    # ─────────────────────────────────────────────────────────────
    # 3. plotting with available variables
    # ─────────────────────────────────────────────────────────────

    layers = visual(Scatter, alpha = 0.05)
    plt = data(df) * layers * mapping(xvars, yvars, col = dims(1) => renamer(string.(yvars) .* "\n" .* NSE_df.NSE_label))
    plt *= mapping(color = dims(1) => renamer(string.(yvars))=> "Obs vs Pred")

    # Calculate number of rows and columns for facet layout
    n_plots = length(yvars)
    
    draw(plt 
        ,axis=(aspect=1, limits = ((0, ymax_list[1]), (0, ymax_list[1]))),
        figure = (size = (400 * n_plots, 400),),
        facet = (; linkxaxes = :none, linkyaxes = :none)
    )
end