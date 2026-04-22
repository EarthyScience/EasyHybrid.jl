import Base.max, Base.min

struct HParameter{T1::Symbol, T2}
    name::T1
    default::T2
    lo::T2
    hi::T2
end

Base.min(hp::HParameter) = hp.lo
Base.max(hp::HParameter) = hp.hi


struct HybridModelParameters{T1<:Vector{HParameter}}
    parameters::T1
end

Base.min(hp::HybridModelParameters) = Base.min.(hp.parameters)
Base.max(hp::HybridModelParameters) = Base.max.(hp.parameters)


struct HybridModel{TNNs<:Tuple{Lux.Chain}, VS<:Vector{Symbol}, M<:Function, HP <: HybridModelParameters} <: LuxCore.AbstractLuxContainerLayer{(:NN,)}
    """
    A tuple of neural networks (NN) that will be used in the hybrid model. Each NN can be a Lux.Chain or any other type of neural network that is compatible with Lux.
    """
    NN::TNNs
    
    """
    A vector of symbols representing the names of the predictors (input features) that will be used in the hybrid model. These names should correspond to the columns in the input data that will be fed into the model.
    """
    predictors_names::VS

    """
    A vector of symbols representing the names of the forcing terms (external inputs) that will be used in the hybrid model. These names should correspond to the columns in the input data that will be fed into the model.
    """
    forcing_names::VS

    """
    A vector of symbols representing the names of the targets (output variables) that will be used in the hybrid model. These names should correspond to the columns in the target data.
    """
    targets_names::VS

    """
    A function that represents the mechanistic model. This function should take the predictors and the parameters as input and return the output of the mechanistic model. The function should be defined in such a way that it can be easily integrated with the neural network components of the hybrid model.
    """
    mechanistic_model::M

    """
    A struct that contains the parameters of the hybrid model. This struct should include all the parameters that are needed for both the mechanistic and neural network components of the model. The parameters should be defined in a way that allows for easy optimization and tuning during the training process.
    """
    parameters::HP

    """
    A vector of symbols representing the names of the parameters that will be used in the neural network.
    """
    neural_parameters::VS

    """
    A vector of symbols representing the names of the global parameters that will be used in the hybrid model.
    """
    global_parameters::VS

    """
    A vector of symbols representing the names of the fixed parameters that will be used in the hybrid model.
    """
    fixed_parameters::VS

    """
    Whether to scale the outputs of the neural network components of the hybrid model. If `scale_nn_outputs` is set to `true`, the outputs of the neural networks will be scaled to a specific range (e.g., between 0 and 1) before being combined with the outputs of the mechanistic model. This can help in ensuring that the contributions of the neural networks are appropriately balanced with those of the mechanistic model during training and inference.
    """
    scale_nn_outputs::Bool

    """
    Whether to start the training of the hybrid model from the default parameters or to initialize the parameters randomly. If `start_from_default` is set to `true`, the training will begin with the default parameter values specified in the `parameters` struct. If it is set to `false`, the parameters will be initialized randomly, which can help in exploring a wider parameter space during training.
    """
    start_from_default::Bool

    """
    Additional arguments to be passed to the neural network components of the hybrid model. This can include any additional settings or configurations that are needed for the neural networks, such as activation functions, regularization terms, or optimization settings.
    """
    config::NamedTuple
end

function HybridModel(predictors_names, forcing_names, targets_names, mechanistic_model, parameters, neural_parameters, global_parameters, fixed_parameters, scale_nn_outputs, start_from_default, config)
    return HybridModel(NN, predictors_names, forcing_names, targets_names, mechanistic_model, parameters, neural_parameters, global_parameters, fixed_parameters, scale_nn_outputs, start_from_default, config)
end