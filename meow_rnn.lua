require 'torch'
require 'nn'
require 'nngraph'

RNN = {}
function RNN.create(opt)
    local x = nn.Identity()()
    local h_prev = nn.Identity()()

    local Wx = nn.Linear(opt.size,opt.size)({x})
    local Uh = nn.Linear(opt.size,opt.size)({h_prev})
    local sum = nn.CAddTable()({Wx,Uh})
    local h = nn.Tanh()({sum})
    
    local y = nn.SoftMax()({nn.Linear(opt.size,opt.size)({h})})

    return nn.gModule({x,h_prev},{h,y})

end



