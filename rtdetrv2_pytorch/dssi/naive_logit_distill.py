import torch
from torch import nn

def distill_one_epoch(teacher_model, student_model, cfg):
    train_loader = cfg.train_dataloader
    N = len(train_loader)
    num_batches_for_cal = int(0.1 * N)

    objective = nn.MSELoss()
    # TODO try the config optimizer since the different net components need different LRs and lumping all the params into a single group didnt seem to work
    optimizer = cfg.student_optimizer(student_model) # TRY THIS
    #optimizer = torch.optim.Adam(student_model.parameters()) # THIS IS WHAT I WAS TRYING AND IT DIDNT WORK

    device = torch.device('mps')

    teacher_model = teacher_model.eval()

    for ind, (imgs, labels) in enumerate(train_loader):
        if ind == num_batches_for_cal:
            break
        optimizer.zero_grad()

        imgs = imgs.to(device)

        with torch.no_grad():
            teacher_output = teacher_model(imgs)

        student_output = student_model(imgs, labels)

        student_logits = student_output['pred_logits']
        student_boxes = student_output['pred_boxes']
        teacher_logits = teacher_output['pred_logits']
        teacher_boxes = teacher_output['pred_boxes']

        logits_loss = objective(student_logits, teacher_logits)
        boxes_loss = objective(student_boxes, teacher_boxes)

        # NOTE weights are taken from OG setcriterion rtdetr config
        loss = 1 * logits_loss + 5 * boxes_loss
        loss.backward()
        optimizer.step()

        if ind % 10 == 0:
            print(f'{ind} / {num_batches_for_cal}', loss.item(), logits_loss.item(), boxes_loss.item())
