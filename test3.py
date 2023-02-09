from accelerate import Accelerator
import torch
import torch.nn as nn


# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin1 = nn.Linear(10, 20)
#         self.lin2 = nn.Linear(20, 10)

#     def forward(self, x):
#         return self.lin2(self.lin1(x))

# accelerator = Accelerator()

# myModel = MyModel()

# model_ddp = accelerator.prepare(myModel)
# print(list(myModel.parameters())[0].device)

# class Test():
#     def __init__(self, model):
#         self.model = model


# test = Test(model_ddp)
# model_ddp.new_attr = 'das'
# print(test.model == model_ddp)
# print(test.model.new_attr)

# x = torch.tensor([2.0], requires_grad=False)
# y = torch.tensor([3.0], requires_grad=True)
# with torch.no_grad():
#     z = torch.tensor([2.0], requires_grad=False)

# print(x.is_leaf)
# print(y.is_leaf)
# print(z.is_leaf)

# with torch.inference_mode():
#     x = torch.randn(1)
#     y = x + 1

# x + torch.tensor([1.3], requires_grad=False)

test = [1,2,3,4,5]
test2 = [6,7,8,9,10]

for idx, (t1, t2) in enumerate(zip(test, test2)):
    print(t1)
    print(t2)