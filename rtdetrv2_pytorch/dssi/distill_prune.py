import os
os.environ.update({'PYTORCH_ENABLE_MPS_FALLBACK': '1'})
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))

from src.core import YAMLConfig
import typer
import torch

from src.data import get_coco_api_from_dataset
from src.solver.det_engine import evaluate as test
from src.solver.det_engine import train_one_epoch

import copy
import torch_pruning as tp

from naive_logit_distill import distill_one_epoch

app = typer.Typer()


def construct_pruner(model, example_inputs, prune_amount):
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    ignored_layers = []
    for n, m in model.named_modules():
        if 'decoder.input_proj' in n: # need
            ignored_layers.append(m)
        if 'decoder.decoder.layers' in n: # need TODO break this down more or figure out if this has something to do w/ num_heads
            ignored_layers.append(m)
        if 'decoder.dec_bbox_head' in n:
            ignored_layers.append(m)
        if 'decoder.dec_score_head' in n:
            ignored_layers.append(m)
        if 'encoder.encoder' in n: # get rid of the early encoder layers since they have the positional encoding that is not a parameter and therefore interferes w/ the pruning
            ignored_layers.append(m)

    iterative_steps = 1#5
    pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=imp, # importance criterion for parameter selection
            iterative_steps=iterative_steps, # the number of iterations to achieve target ratio
            pruning_ratio=prune_amount, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            #root_module_types=(nn.Linear, nn.LayerNorm),
            #head_pruning_ratio=0.5,
            isomorphic=True,
            global_pruning=True,
    )

    return pruner, iterative_steps

@app.command()
def main(
    output_path: str,
    prune_amount: float,
    eval_baseline: bool = False,
    eval_pruned: bool = True,
    distill_flag: bool = True,
):
    cfg_path = '/Users/personbear/Documents/DSSI_2024/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco.yml'
    resume_path = '/Users/personbear/Documents/DSSI_2024/RT-DETR/rtdetrv2_pytorch/pretrained_models/rtdetrv2_r18vd_sp1_120e_coco.pth'
    pretrained_model_path = '/Users/personbear/Documents/DSSI_2024/RT-DETR/rtdetrv2_pytorch/pretrained_models/pretrained_model_extracted.pth'

    cfg = YAMLConfig(
        cfg_path,
        resume=resume_path,
    )

    model = cfg.model

    
    state = torch.load(pretrained_model_path, map_location='cpu')
    model.load_state_dict(state['ema']['module'])

    device = torch.device("mps")

    test_dataloader = cfg.val_dataloader
    base_ds = get_coco_api_from_dataset(test_dataloader.dataset)

    criterion = cfg.criterion
    postprocessor = cfg.postprocessor
    evaluator = cfg.evaluator

    model = model.to(device)
    criterion = criterion.to(device)

    if eval_baseline:
        test_stats, coco_evaluator = test(
            model,
            criterion,
            postprocessor,
            test_dataloader,
            evaluator,
            device,
        )

    pruned_model = copy.deepcopy(model)
    pruned_model.eval()

    example_inputs = torch.zeros((1, 3, 640, 640)).to(device)
    pruner, iterative_steps = construct_pruner(pruned_model, example_inputs, prune_amount)

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print('base', base_macs, base_nparams)
    for i in range(iterative_steps):
        # 3. the pruner.step will remove some channels from the model with least importance
        pruner.step()

        # 4. Do whatever you like here, such as fintuning
        macs, nparams = tp.utils.count_ops_and_params(pruned_model, example_inputs)
        print(f'step {i}', macs, nparams)
        if distill_flag:
            distill_one_epoch(model, pruned_model, cfg)

    del pruner

    if eval_pruned:
        test_stats, coco_evaluator = test(
            pruned_model,
            criterion,
            postprocessor,
            test_dataloader,
            evaluator,
            device,
        )


    output_dir = osp.dirname(output_path)
    if not osp.exists(output_dir):
        print(f"Making output directory {output_dir} since it does not exist")
        os.makedirs(output_dir)


    output = {
        'ema': {
            'module': pruned_model.state_dict()
        },
    }
    torch.save(output, output_path)


if __name__ == "__main__":
    app()
