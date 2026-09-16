import torch
from model import Ruri130mClapProjectionModel

model = Ruri130mClapProjectionModel()
state_dict = torch.load(".var/output/model_130m.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 512)
torch.onnx.export(
    model,
    (dummy_input,),
    ".var/output/model_130m.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    external_data=False,
)
