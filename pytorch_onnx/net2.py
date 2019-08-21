import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

PATH = Path('models')
PATH.mkdir(parents=True,exist_ok=True)


dummy_input = torch.randn(1, 3, 32, 32)
input_names = ['input']
output_names = ['output']

#torch.onnx.export(model, dummy_input, "model1.onnx", verbose=True, input_names=input_names, output_names=output_names)
traced_model = torch.jit.trace(model, dummy_input)
#torch.jit.save(traced_model, 'model1.pt')
from ml2rt import save_torch
save_torch(traced_model, 'model1.pt')

torch.onnx.export(model, dummy_input, "model1.onnx",  input_names=input_names, output_names=output_names)

import onnx

# Load the ONNX model
model_onnx = onnx.load("model1.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model_onnx)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model_onnx.graph)

from ml2rt import save_onnx
save_onnx(model_onnx, "model1_d.onnx")
