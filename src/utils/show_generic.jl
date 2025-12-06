function _print_field(io::IO, key::String, value; key_color = :light_black, value_color = :cyan)
    printstyled(io, "  ", key, color = key_color)
    print(io, " = ")

    if isa(value, NamedTuple)
        fields = fieldnames(typeof(value))
        print(io, "(; ")
        for (i, fname) in enumerate(fields)
            fval = getfield(value, fname)
            # Determine color for the field value
            fcolor = if isa(fval, Bool)
                fval ? :green : :red
            elseif isa(fval, Number)
                value_color
            elseif isa(fval, Function)
                :light_blue
            else
                value_color
            end
            # Convert value to string (functions by name)
            fstr = isa(fval, Function) ? nameof(fval) : repr(fval)
            printstyled(io, string(fname), color = :light_black)
            print(io, " = ")
            printstyled(io, fstr, color = fcolor)
            if i < length(fields)
                print(io, ", ")
            else
                print(io, ",")
            end
        end
        print(io, ")")
    else
        # Determine color for single value
        vcolor = if isa(value, Bool)
            value ? :green : :red
        elseif isa(value, Number)
            value_color
        elseif isa(value, Function)
            :light_blue
        else
            value_color
        end
        # Convert function to name, others via repr
        vstr = isa(value, Function) ? nameof(value) : repr(value)
        printstyled(io, vstr, color = vcolor)
    end

    return println(io)
end

function _print_header(io::IO, text::String; color = :blue, bold = true)
    printstyled(io, text, color = color, bold = bold)
    return println(io)
end
function Base.show(io::IO, ::MIME"text/plain", hp::HybridParams)
    _print_header(io, "Hybrid Parameters", color = :blue)

    # Delegate to the contained ParameterContainer
    io_full = IOContext(IndentedIO(io), :compact => false, :limit => false)
    return show(io_full, MIME"text/plain"(), hp.hybrid)
end

function Base.show(io::IO, hp::HybridParams)
    print(io, "HybridParams(")
    show(io, hp.hybrid)
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pc::ParameterContainer)
    table = pc.table
    return PrettyTables.pretty_table(
        io, table;
        column_labels = collect(keys(table.axes[2])),
        row_labels = collect(keys(table.axes[1])),
        alignment = :r,
    )
end

# compact show for nested usage
function Base.show(io::IO, pc::ParameterContainer)
    print(io, "ParameterContainer(")
    print(io, join([string(k) for k in keys(pc.values)], ", "))
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", hm::SingleNNHybridModel)
    _print_header(io, "Hybrid Model (Single NN)")

    printstyled(io, "Neural Network: \n", color = :light_black)
    show(IndentedIO(io), MIME"text/plain"(), hm.NN)
    println(io)

    _print_header(io, "Configuration:", color = :light_blue, bold = false)
    _print_field(io, "predictors", hm.predictors)
    _print_field(io, "forcing", hm.forcing)
    _print_field(io, "targets", hm.targets)
    _print_field(io, "mechanistic_model", hm.mechanistic_model, value_color = :light_blue)
    _print_field(io, "neural_param_names", hm.neural_param_names, value_color = :light_blue)
    _print_field(io, "global_param_names", hm.global_param_names, value_color = :green)
    _print_field(io, "fixed_param_names", hm.fixed_param_names, value_color = :yellow)
    _print_field(io, "scale_nn_outputs", hm.scale_nn_outputs, value_color = hm.scale_nn_outputs ? :green : :red)
    _print_field(io, "start_from_default", hm.start_from_default, value_color = hm.start_from_default ? :green : :red)
    _print_field(io, "config", hm.config, value_color = :cyan)

    println(io)
    _print_header(io, "Parameters:", color = :light_blue, bold = false)
    io_full = IOContext(IndentedIO(io), :compact => false, :limit => false)
    return show(io_full, MIME"text/plain"(), hm.parameters)
end

function Base.show(io::IO, ::MIME"text/plain", hm::MultiNNHybridModel)
    _print_header(io, "Hybrid Model (Multi NN)")

    # Neural networks
    _print_header(io, "Neural Networks:", color = :light_black, bold = false)
    n_nns = length(hm.NNs)
    idx = 0
    for (name, nn) in pairs(hm.NNs)
        idx += 1
        printstyled(io, "  ", color = :light_black)
        printstyled(io, string(name), color = :cyan, bold = true)
        println(io, ":")
        show(IndentedIO(io; indent = "    "), MIME"text/plain"(), nn)
        if idx < n_nns
            println(io)
        end
    end
    println(io)

    # Configuration
    _print_header(io, "Configuration:", color = :light_blue, bold = false)

    # Predictors are per-network (NamedTuple)
    printstyled(io, "  predictors", color = :light_black)
    println(io, ":")
    for (name, preds) in pairs(hm.predictors)
        printstyled(io, "    ", color = :light_black)
        printstyled(io, string(name), color = :yellow)
        print(io, " = ")
        printstyled(io, preds, color = :cyan)
        println(io)
    end

    _print_field(io, "forcing", hm.forcing)
    _print_field(io, "targets", hm.targets)
    _print_field(io, "mechanistic_model", hm.mechanistic_model, value_color = :light_blue)
    _print_field(io, "neural_param_names", hm.neural_param_names, value_color = :light_blue)
    _print_field(io, "global_param_names", hm.global_param_names, value_color = :green)
    _print_field(io, "fixed_param_names", hm.fixed_param_names, value_color = :yellow)
    _print_field(
        io, "scale_nn_outputs", hm.scale_nn_outputs,
        value_color = hm.scale_nn_outputs ? :green : :red
    )
    _print_field(
        io, "start_from_default", hm.start_from_default,
        value_color = hm.start_from_default ? :green : :red
    )
    _print_field(io, "config", hm.config, value_color = :cyan)

    println(io)

    # Parameters
    _print_header(io, "Parameters:", color = :light_blue, bold = false)
    io_full = IOContext(IndentedIO(io), :compact => false, :limit => false)
    return show(io_full, MIME"text/plain"(), hm.parameters)
end

mutable struct IndentedIO{IOType <: IO} <: IO
    io::IOType
    indent::String
    at_line_start::Bool
end

function IndentedIO(io::IO; indent = "  ")
    return IndentedIO{typeof(io)}(io, indent, true)
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