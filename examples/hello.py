# import numpy as np
# from tinygrad.nn.optim import Adam
# from tinygrad.tensor import Tensor

# class TinyNet:
#   def __init__(self):
#     x_init = np.random.randn(1,4).astype(np.float32)
#     W_init = np.random.randn(4,4).astype(np.float32)
#     m_init = np.random.randn(1,4).astype(np.float32)

#     self.x = Tensor(x_init.copy(), requires_grad=True)
#     self.W = Tensor(W_init.copy(), requires_grad=True)
#     self.m = Tensor(m_init.copy())

#   def forward(self):
#     out = self.x.matmul(self.W).relu()
#     # print(out.detach().numpy())
#     out = out.log_softmax(1)
#     out = out.mul(self.m).add(self.m).sum()
#     return out

# def step(steps=1, kwargs={}):
#   net = TinyNet()
#   optim = Adam([net.x, net.W], **kwargs)
#   for _ in range(steps):
#     out = net.forward()
#     optim.zero_grad()
#     out.backward()
#     optim.step()
#   return net.x.detach().numpy(), net.W.detach().numpy()

# net = TinyNet()
# out = net.forward()

# from tinygrad.jit import CacheCollector
# CacheCollector.start()       # enables the cache
# out.realize()             # create the program and runs it
# cache_saved = CacheCollector.finish()  # disable the cache

# # there's one ASTRunner in the cache
# # assert len(cache_saved) == 1

# # print the C Program :)
# print(cache_saved[0].prg.prg)

# # print(step())


# --------------------
# addition example

# from tinygrad.tensor import Tensor
# from tinygrad.jit import CacheCollector

# breakpoint()
# result = Tensor(2) + Tensor(3)
# CacheCollector.start()       # enables the cache
# result.realize()             # create the program and runs it
# cache_saved = CacheCollector.finish()  # disable the cache

# # print the C Program :)
# print(cache_saved[0].prg.prg)


#---------------
# bug repro
# 1. fix the bug in tinygrad cache_saved[0].prg.prg doesnt work with DEBUG=5
from tinygrad.tensor import Tensor
from tinygrad.jit import CacheCollector

import os
breakpoint()

a = Tensor(2)
b = Tensor(3)
result = a+b

CacheCollector.start()       # enables the cache
result.realize()             # create the program and runs it
cache_saved = CacheCollector.finish()  # disable the cache

# print the C Program :)
print(cache_saved[0].prg.prg)






"""
TODO: 
    1. fix the bug in tinygrad cache_saved[0].prg.prg doesnt work with DEBUG=5
    2. when opencl ran the kernel how did it passed the info to result tensor?
        rawbuffer should hold the value, but it was never passed to result tensor, but result.numpy() gives value back
    3. debug again cpu and conv kernel
"""
