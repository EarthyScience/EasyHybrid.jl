function getWdata(xin, y_target; ws = 8)
    features = size(xin)[1]
    xdata = slidingwindow(xin, ws, stride = 1)
    ydata = y_target[ws:length(xdata)+ws-1]
    xwindowed = zeros(Float32, ws, features, length(ydata))
    for i in 1:length(ydata)
        xwindowed[:, :, i] = getobs(xdata, i)'
    end
    return xwindowed, ydata
end