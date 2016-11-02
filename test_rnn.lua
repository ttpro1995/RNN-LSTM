require 'meow_rnn'
opt = {}
opt.size = 4
m = RNN.create(opt)

function test_eval()

    local x = torch.Tensor{1,2,3,4}
    local h = torch.Tensor{5,6,7,8}

    local ans = m:forward({x,h})
    local h_next = ans[1]
    local y_hat = ans[2]
    
    return ans, h_next, y_hat
end

