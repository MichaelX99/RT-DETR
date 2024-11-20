import torch
from torch import nn

class DETRDistill(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()

        self.teacher = teacher_model
        self.student = student_model

        self._setup_hooks()

        # Since the student network can be smaller than the teacher network, run inference with a fake input to get their shapes
        fake_input = torch.zeros(1, 3, 640, 640).to('mps')
        with torch.no_grad():
            self.teacher(fake_input)
            self.student(fake_input)
        student_encoder_features, teacher_encoder_features = self.get_student_teacher_encoder_features()
        teacher_encoder_channels = [x.shape[1] for x in teacher_encoder_features]
        student_encoder_channels = [x.shape[1] for x in student_encoder_features]

        # Form the learned projection
        self.encoder_projection = nn.ModuleList([
            nn.Linear(student_ch, teacher_ch) for (student_ch, teacher_ch) in zip(student_encoder_channels, teacher_encoder_channels)
        ])

    def _setup_hooks(self):
        # Create hooks to be able to access the intermediate encoder outputs during a forward call
        self.teacher_encoder_features = None
        def teacher_encoder_hook(module, input, output):
            self.teacher_encoder_features = output
            return output
        self.teacher.encoder.register_forward_hook(teacher_encoder_hook)

        # Create hooks to be able to access the intermediate encoder outputs during a forward call
        self.student_encoder_features = None
        def student_encoder_hook(module, input, output):
            self.student_encoder_features = output
            return output
        self.student.encoder.register_forward_hook(student_encoder_hook)

    def get_student_teacher_encoder_features(self):
        # Get the output encoder features from the teacher network
        teacher_encoder_features = self.teacher_encoder_features
        self.teacher_encoder_features = None

        # Get the output encoder features from the student network
        student_encoder_features = self.student_encoder_features
        self.student_encoder_features = None

        return student_encoder_features, teacher_encoder_features

    def forward(self, x):
        output = {}
        with torch.no_grad():
            teacher_output = self.teacher(x)

        student_output = self.student(x)

        student_encoder_features, teacher_encoder_features = self.get_student_teacher_encoder_features()

        projected_student_encoder_features = []
        for proj_layer, student_feat in zip(self.encoder_projection, student_encoder_features):
            reshaped_feat = student_feat.permute(0, 2, 3, 1) # change the axis so the channel dim is the last one
            proj_feat = proj_layer(reshaped_feat)
            proj_feat = proj_feat.permute(0, 3, 1, 2) # change the axis so that the channel dimension is again the 1th dimension
            projected_student_encoder_features.append(proj_feat)

        output['teacher_encoder_features'] = teacher_encoder_features
        output['student_encoder_features'] = projected_student_encoder_features

        return output

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

    distiller = DETRDistill(teacher_model, student_model).to(device)

    for ind, (imgs, labels) in enumerate(train_loader):
        if ind == num_batches_for_cal:
            break
        optimizer.zero_grad()

        imgs = imgs.to(device)

        output = distiller(imgs)

        breakpoint()

