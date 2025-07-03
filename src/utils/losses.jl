export lossfn
using Lux

"""
    lossfn(lhm::LinearHM, ds, (y, no_nan), ps, st)
"""
function lossfn(lhm::LinearHM, ds, (y, no_nan), ps, st)
    ŷ, αst = lhm(ds, ps, st)
    _, st = αst
    loss = mean((y[no_nan] .- ŷ[no_nan]).^2)
    return loss
end


"""
    lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st)
"""
function lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)

    loss = 0.0
    for k in axiskeys(y, 1)
        loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
    end
    return loss
end

"""
    lossfn(HM::RespirationRbQ10, ds, y, ps, st)
"""
function lossfn(HM::BulkDensitySOC, ds_p, (ds_t, ds_t_nan), ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)
    ŷ, _ = HM(ds_p, ps, st)

    loss = 0.0
    for k in axiskeys(y, 1)
        loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
    end    
    return loss
end

"""
    lossfn(NN::NaiveNN, ds, y, ps, st)
"""
function lossfn(NN::Lux.Chain, ds_p, (ds_t, ds_t_nan), ps, st)
    ŷ, _ = NN(ds_p, ps, st) 
    y = Matrix(ds_t) 

    diff2 = (ŷ .- y).^2
    rmse  = sqrt(mean(diff2[ds_t_nan]))
    return rmse
end

"""
    lossfn(mh::MultiHeadNN, ds, y, ps, st)
"""
function lossfn(mh::MultiHeadNN, ds_p, (ds_t, ds_t_nan), ps, st)
    ŷ, _ = mh(ds_p, ps, st)     

    loss  = 0.0
    nkeys = 0
    for k in 1:size(ŷ,1)        
        idx = ds_t_nan[k, :]              
        nk  = count(idx)
        if nk > 0
            loss += mean((ŷ[k,idx] .- ds_t[k,idx]).^2)
            nkeys += 1
        end
    end
    rmse = sqrt(loss / nkeys)          
    return rmse
end
