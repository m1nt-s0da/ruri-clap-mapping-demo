import torch
from model import RuriClapProjectionModel

model = RuriClapProjectionModel()
state_dict = torch.load(".var/output/model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 256)
torch.onnx.export(
    model,
    (dummy_input,),
    ".var/output/model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    external_data=False,
)
