require 'torch'
require 'nn'
require 'nngraph'

function create_lstm_cell(opt)
    local x_t = nn.Identity()()
    local h_prev = nn.Identity()()
    local c_prev = nn.Identity()()
    
    
    function input_sum()
        local W_x = nn.Linear(opt.size, opt.size)({x_t})
        local U_h_prev = nn.Linear(opt.size, opt.size)({h_prev})
        -- TODO: bias b ?
        local sum = nn.CAddTable()({W_x,U_h_prev})
        return sum
    end

   local i = nn.Sigmoid()({input_sum()})
   local f = nn.Sigmoid()({input_sum()})
   local o = nn.Sigmoid()({input_sum()})
   local u = nn.Tanh()({input_sum()})

   local i_dot_u = nn.CMulTable()({i,u})
   local f_dot_c_prev = nn.CMulTable()({f,c_prev})
   local c = nn.CAddTable()({i_dot_u,f_dot_c_prev})

   local tanh_c = nn.Tanh()({c})
   local h = nn.CMulTable()({o,tanh_c})
  return nn.gModule({x_t,h_prev,c_prev},{h,c})
   


end


