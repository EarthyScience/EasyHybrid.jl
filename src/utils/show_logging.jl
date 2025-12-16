function _format_loss_spec(io::IO, spec::SymbolicLoss)
    printstyled(io, ":", color = :light_blue)
    return printstyled(io, spec.name, color = :cyan)
end

function _format_loss_spec(io::IO, spec::FunctionLoss)
    return printstyled(io, nameof(spec.f), color = :light_blue)
end

function _format_loss_spec(io::IO, spec::ParameterizedLoss)
    printstyled(io, nameof(spec.f), color = :light_blue)
    return if !isempty(spec.args) && !isempty(spec.kwargs)
        print(io, "(")
        printstyled(io, join(spec.args, ", "), color = :yellow)
        print(io, "; ")
        printstyled(io, join(["$k=$v" for (k, v) in pairs(spec.kwargs)], ", "), color = :green)
        print(io, ")")
    elseif !isempty(spec.args)
        print(io, "(")
        printstyled(io, join(spec.args, ", "), color = :yellow)
        print(io, ")")
    elseif !isempty(spec.kwargs)
        print(io, "(; ")
        printstyled(io, join(["$k=$v" for (k, v) in pairs(spec.kwargs)], ", "), color = :green)
        print(io, ")")
    end
end

function _format_loss_spec(io::IO, spec::ExtraLoss)
    return if spec.f === nothing
        printstyled(io, "nothing", color = :light_black)
    else
        printstyled(io, nameof(spec.f), color = :light_blue)
    end
end

function Base.show(io::IO, ::MIME"text/plain", ll::LoggingLoss)
    return if get(io, :compact, false)
        printstyled(io, "LoggingLoss", color = :blue, bold = true)
        print(io, "(")
        _format_loss_spec(io, ll.training_loss)
        print(io, ")")
    else
        printstyled(io, "LoggingLoss", color = :blue, bold = true)
        print(io, "(\n")

        printstyled(io, "  loss_types", color = :light_black)
        print(io, " = [")
        for (i, loss) in enumerate(ll.loss_types)
            _format_loss_spec(io, loss)
            if i < length(ll.loss_types)
                print(io, ", ")
            end
        end
        print(io, "],\n")

        printstyled(io, "  training_loss", color = :light_black)
        print(io, " = ")
        _format_loss_spec(io, ll.training_loss)
        print(io, ",\n")

        printstyled(io, "  extra_loss", color = :light_black)
        print(io, " = ")
        _format_loss_spec(io, ll.extra_loss)
        print(io, ",\n")

        printstyled(io, "  agg", color = :light_black)
        print(io, " = ")
        printstyled(io, nameof(ll.agg), color = :light_blue)
        print(io, ",\n")

        printstyled(io, "  train_mode", color = :light_black)
        print(io, " = ")
        printstyled(io, ll.train_mode, color = ll.train_mode ? :green : :red)
        print(io, "\n)")
    end
end
