require 'torch'
require 'nn'
require 'nngraph'

RNN = {}
function RNN.create(opt)
    local x = nn.Identity()()
    local h_prev = nn.Identity()()

    local Wx = nn.Linear(opt.size,opt.size)({x})
    local Uh_prev = nn.Linear(opt.size,opt.size)({h_prev})
    local sum = nn.CAddTable()({Wx,Uh_prev})
    local h = nn.Tanh()({sum})
    
    local Vh = nn.Linear(opt.size,opt.size)({h})
    local y = nn.LogSoftMax()({Vh})

    return nn.gModule({x,h_prev},{h,y})

end



