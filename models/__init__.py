from models.ecgfounder.loader import load_ecgfounder
from models.ecgfounder.evaluator import evaluate_ecgfounder
from models.ecgfounder.postprocess import postprocess_ecgfounder

from models.xecg.loader import load_xecg
from models.xecg.evaluator import evaluate_xecg
from models.xecg.postprocess import postprocess_xecg

# from models.dsail_snu.loader import load_dsail
# from models.dsail_snu.evaluator import evaluate_dsail
# from models.dsail_snu.postprocess import postprocess_dsail

def load_model(name, device):
    if name.lower() == "ecgfounder":
        return load_ecgfounder(device)
    elif name.lower() == "xecg":
        return load_xecg(device)
    # elif name.lower() == "dsail":
    #     return load_dsail(device)
    raise ValueError("Unknown model")

def get_evaluator(name):
    if name.lower() == "ecgfounder":
        return evaluate_ecgfounder
    elif name.lower() == "xecg":
        return evaluate_xecg
    # elif name.lower() == "dsail":
    #     return evaluate_dsail
    raise ValueError("Unknown model")

def get_postprocessor(name):
    if name.lower() == "ecgfounder":
        return postprocess_ecgfounder
    elif name.lower() == "xecg":
        return postprocess_xecg
    raise ValueError("Unknown model")