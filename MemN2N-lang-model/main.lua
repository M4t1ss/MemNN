-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('xlua')
require('paths')
local tds = require('tds')
paths.dofile('data.lua')
paths.dofile('model.lua')

local function train(words)
    local N = math.ceil(words:size(1) / g_params.batchsize)
    local cost = 0
    local y = torch.ones(1)
	local input
	local target
	local context
	local time
	if g_params.gpu > 0 then
		print('Training on GPU')
		input = torch.CudaTensor(g_params.batchsize, g_params.edim)
		target = torch.CudaTensor(g_params.batchsize)
		context = torch.CudaTensor(g_params.batchsize, g_params.memsize)
		time = torch.CudaTensor(g_params.batchsize, g_params.memsize)
	else
		print('Training on CPU')
		input = torch.Tensor(g_params.batchsize, g_params.edim)
		target = torch.Tensor(g_params.batchsize)
		context = torch.Tensor(g_params.batchsize, g_params.memsize)
		time = torch.Tensor(g_params.batchsize, g_params.memsize)
	end
    input:fill(g_params.init_hid)
    for t = 1, g_params.memsize do
        time:select(2, t):fill(t)
    end
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
            local m = math.random(g_params.memsize + 1, words:size(1)-1)
            target[b] = words[m+1]
            context[b]:copy(
                words:narrow(1, m - g_params.memsize + 1, g_params.memsize))
        end
        local x = {input, target, context, time}
        local out = g_model:forward(x)
        cost = cost + out[1]
        g_paramdx:zero()
        g_model:backward(x, y)
        local gn = g_paramdx:norm()
        if gn > g_params.maxgradnorm then
            g_paramdx:mul(g_params.maxgradnorm / gn)
        end
        g_paramx:add(g_paramdx:mul(-g_params.dt))
    end
    return cost/N/g_params.batchsize
end

local function test(words)
    local N = math.ceil(words:size(1) / g_params.batchsize)
    local cost = 0
	local input
	local target
	local context
	local time
	if g_params.gpu > 0 then
		print('Testing on GPU')
		input = torch.CudaTensor(g_params.batchsize, g_params.edim)
		target = torch.CudaTensor(g_params.batchsize)
		context = torch.CudaTensor(g_params.batchsize, g_params.memsize)
		time = torch.CudaTensor(g_params.batchsize, g_params.memsize)
	else
		print('Testing on CPU')
		input = torch.Tensor(g_params.batchsize, g_params.edim)
		target = torch.Tensor(g_params.batchsize)
		context = torch.Tensor(g_params.batchsize, g_params.memsize)
		time = torch.Tensor(g_params.batchsize, g_params.memsize)
	end
    input:fill(g_params.init_hid)
    for t = 1, g_params.memsize do
        time:select(2, t):fill(t)
    end
    local m = g_params.memsize + 1
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
            target[b] = words[m+1]
            context[b]:copy(
                words:narrow(1, m - g_params.memsize + 1, g_params.memsize))
            m = m + 1
            if m > words:size(1)-1 then
                m = g_params.memsize + 1
            end
        end
        local x = {input, target, context, time}
        local out = g_model:forward(x)
        cost = cost + out[1]
    end
    return cost/N/g_params.batchsize
end

local function save(path)
    local d = {}
    d.params = g_params
    d.paramx = g_paramx:float()
    d.log_cost = g_log_cost
    d.log_perp = g_log_perp
    torch.save(path, d)
end

local function run(epochs)
	if g_params.test then
		local ctt = test(g_words_test)
		print('Perplexity: ' .. math.exp(ctt))
	else
		for i = 1, epochs do
			local c, ct
			if g_params.test~=true then
				c = train(g_words_train)
				ct = test(g_words_valid)

				-- Logging
				local m = #g_log_cost+1
				g_log_cost[m] = {m, c, ct}
				g_log_perp[m] = {m, math.exp(c), math.exp(ct)}
				local stat = {perplexity = math.exp(c) , epoch = m,
						valid_perplexity = math.exp(ct), LR = g_params.dt}
			end

			-- Learning rate annealing
			if m > 1 and g_log_cost[m][3] > g_log_cost[m-1][3] * 0.9999 then
				g_params.dt = g_params.dt / 1.5
				if g_params.dt < 1e-5 then break end
			end
			if g_params.save ~= '' then
				save(g_params.save)
			end
		end
	end
end

--------------------------------------------------------------------
--------------------------------------------------------------------
-- model params:
local cmd = torch.CmdLine()
cmd:option('--gpu', 1, 'GPU id to use')
cmd:option('--edim', 150, 'internal state dimension')
cmd:option('--lindim', 75, 'linear part of the state')
cmd:option('--init_std', 0.05, 'weight initialization std')
cmd:option('--init_hid', 0.1, 'initial internal state value')
cmd:option('--sdt', 0.01, 'initial learning rate')
cmd:option('--maxgradnorm', 50, 'maximum gradient norm')
cmd:option('--memsize', 100, 'memory size')
cmd:option('--nhop', 6, 'number of hops')
cmd:option('--batchsize', 128)
cmd:option('--show', false, 'print progress')
cmd:option('--load', '', 'model file to load')
cmd:option('--save', '', 'path to save model')
cmd:option('--epochs', 100)
cmd:option('--test', false, 'enable testing')
cmd:option('--data_name', '', 'enable testing')
g_params = cmd:parse(arg or {})

if g_params.test~=true then
	print(g_params)
end
if g_params.gpu > 0 then
	print('Using GPU ' .. g_params.gpu)
	cutorch.setDevice(g_params.gpu)
end

g_vocab =  tds.hash()
g_ivocab =  tds.hash()
g_ivocab[#g_vocab+1] = '<eos>'
g_vocab['<eos>'] = #g_vocab+1

g_words_train = g_read_words(g_params.data_name .. '.train.txt', g_vocab, g_ivocab) --/home/matiss/data/dgt
g_words_valid = g_read_words(g_params.data_name .. '.valid.txt', g_vocab, g_ivocab)
g_words_test = g_read_words(g_params.data_name .. '.test.txt', g_vocab, g_ivocab)
g_params.nwords = #g_vocab
if g_params.test~=true then
	print('vocabulary size ' .. #g_vocab)
end

g_model = g_build_model(g_params)
g_paramx, g_paramdx = g_model:getParameters()
g_paramx:normal(0, g_params.init_std)
if g_params.load ~= '' then
    local f = torch.load(g_params.load)
    g_paramx:copy(f.paramx)
end

g_log_cost = {}
g_log_perp = {}
g_params.dt = g_params.sdt

if g_params.test~=true then
	print('starting to run....')
end
run(g_params.epochs)

if g_params.save ~= '' then
    save(g_params.save)
end
