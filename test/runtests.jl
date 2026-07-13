using EasyHybrid
using Test

# Include GenericHybridModel tests
include("test_generic_hybrid_model.jl")
# Include SplitData tests
include("test_split_data_train.jl")
include("test_autodiff_backend.jl")
include("test_loss_types.jl")
include("test_show_loss_types.jl")
include("test_compute_loss.jl")
include("test_loss_fn.jl")
include("test_show_train.jl")
include("test_show_generic_hybrid.jl")
include("test_wrap_tuples.jl")
include("test_extract_weights.jl")
include("test_transformers.jl")
include("test_seq2seq.jl")
