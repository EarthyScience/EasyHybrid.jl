# show methods
function print_key_value(io, key, value; color=:light_red)
    print(io, "$key: ")
    printstyled(io, value; color=color) # TODO: counts the corresponding number of trainable parameters
    println(io)
end

function Base.show(io::IO, pc::ParameterContainer)
    table = pc.table
    println(io)
    PrettyTables.pretty_table(
        io, table;
        header     = collect(keys(table.axes[2])),
        row_labels = collect(keys(table.axes[1])),
        alignment  = :r,
    )
end

function Base.show(io::IO, hm::SingleNNHybridModel)
    println(io, "Neural Network:")
    show(IndentedIO(io), MIME"text/plain"(), hm.NN)
    println(io)
    
    print_key_value(io, "Predictors", hm.predictors)
    print_key_value(io, "Forcing", hm.forcing)
    print_key_value(io, "Neural parameters", hm.neural_param_names)
    print_key_value(io, "Global parameters", hm.global_param_names)
    print_key_value(io, "Fixed parameters", hm.fixed_param_names)
    print_key_value(io, "Scale NN outputs", hm.scale_nn_outputs)
    
    println(io, "Parameter defaults and bounds:")
    show(IndentedIO(io), MIME"text/plain"(), hm.parameters)
end

function Base.show(io::IO, hm::MultiNNHybridModel)
    printstyled(io, "Neural Networks:"; color=:light_yellow)
    println(io)
    for (name, nn) in pairs(hm.NNs)
        printstyled(io, "$name:\n"; color=:light_black)
        show(IndentedIO(io), MIME"text/plain"(), nn)
        println(io)  # add spacing between networks
    end
    
    print(io, "Predictors:")
    for (name, preds) in pairs(hm.predictors)
        printstyled(io, " $name: ", color=:light_black)
        printstyled(io, preds; color=:light_red)
        println(io)
    end
    
    print_key_value(io, "Forcing", hm.forcing)
    print_key_value(io, "Neural parameters", hm.neural_param_names)
    print_key_value(io, "Global parameters", hm.global_param_names)
    print_key_value(io, "Fixed parameters", hm.fixed_param_names)
    print_key_value(io, "Scale NN outputs", hm.scale_nn_outputs)
    
    println(io, "Parameter defaults and bounds:")
    show(IndentedIO(io), MIME"text/plain"(), hm.parameters)
end

mutable struct IndentedIO{IOType<:IO} <: IO
    io::IOType
    indent::String
    at_line_start::Bool
end

function IndentedIO(io::IO; indent="    ")
    IndentedIO{typeof(io)}(io, indent, true)
end

function Base.write(ido::IndentedIO, data::UInt8)
    c = Char(data)
    if ido.at_line_start && c != '\n'
        write(ido.io, ido.indent)
        ido.at_line_start = false
    end
    write(ido.io, data)
    ido.at_line_start = (c == '\n')
    return 1
end

Base.flush(ido::IndentedIO) = flush(ido.io)
Base.isopen(ido::IndentedIO) = isopen(ido.io)
Base.close(ido::IndentedIO) = close(ido.io)
Base.readavailable(ido::IndentedIO) = readavailable(ido.io)