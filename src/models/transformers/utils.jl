import ..EasyHybrid: get_layer_dim

function get_layer_dim(l::FeatureEmbedding, ::Val{:input})
    return get_layer_dim(l.dense, Val{:input}())
end
function get_layer_dim(l::PatchEmbedding, ::Val{:input})
    return get_layer_dim(l.conv, Val{:input}())
end

function get_layer_dim(l::TransformerModel, ::Val{:input})
    return l.stem isa NoOpLayer ? get_layer_dim(l.embedding, Val{:input}()) : get_layer_dim(l.stem, Val{:input}())
end
function get_layer_dim(l::TransformerModel, ::Val{:output})
    return get_layer_dim(l.output, Val{:output}())
end

function get_layer_dim(l::Union{VisionTransformer, VisionToVisionModel}, ::Val{:input})
    return l.stem isa NoOpLayer ? get_layer_dim(l.patch_embedding, Val{:input}()) : get_layer_dim(l.stem, Val{:input}())
end
function get_layer_dim(l::Union{VisionTransformer, VisionToVisionModel}, ::Val{:output})
    return get_layer_dim(l.output, Val{:output}())
end

function get_layer_dim(l::EncoderDecoderModel, ::Val{:input})
    return l.stem isa NoOpLayer ? get_layer_dim(l.enc_embedding, Val{:input}()) : get_layer_dim(l.stem, Val{:input}())
end
function get_layer_dim(l::EncoderDecoderModel, ::Val{:output})
    return get_layer_dim(l.output, Val{:output}())
end
