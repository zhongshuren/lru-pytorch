import torch
from tqdm import tqdm
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


def get_optimizers(args, model):
    regular_params = []
    ssm_params = []
    for name, param in model.named_parameters():
        if 'B' in name or 'params' in name or 'norm' in name:
            ssm_params.append(param)
        else:
            regular_params.append(param)

    optimizer_regular = AdamW(regular_params, lr=args.lr_base, weight_decay=args.weight_decay)
    optimizer_ssm = Adam(ssm_params, lr=args.lr_base * args.lr_factor)

    scheduler_regular = CosineAnnealingLR(optimizer_regular, T_max=args.epochs, eta_min=args.lr_min)
    scheduler_ssm = CosineAnnealingLR(optimizer_ssm, T_max=args.epochs, eta_min=args.lr_min)
    return (optimizer_regular, optimizer_ssm), (scheduler_regular, scheduler_ssm)


def create_mask(label, real_lengths):
    mask = torch.arange(label.shape[1]).repeat(label.shape[0], 1).to(label.device)
    mask = torch.logical_and(mask.ge(real_lengths[:, :1]), mask.lt(real_lengths[:, 1:]))
    return mask


def cross_entropy(logits, label):
    log_softmax = torch.log_softmax(logits, dim=-1)
    one_hot = F.one_hot(label, num_classes=2)
    return -torch.sum(log_softmax * one_hot, dim=-1)


def calc_accuracy(logits, label, mask):
    pred = torch.argmax(logits, dim=-1).detach()
    return (torch.eq(pred, label).float().mean(dim=-1) * mask).sum().item()


def step(dataloader, model, device, optimizers=None, mode='train'):
    loss_list, acc_cnt, tot_cnt = [], 0, 0
    for i, (input_seq, label, real_lengths) in tqdm(enumerate(dataloader), desc=mode):
        input_seq = input_seq.to(device)
        label = label.to(device).long()
        real_lengths = real_lengths['lengths'].to(device)
        mask = create_mask(label, real_lengths)
        logits = model(input_seq)

        losses = cross_entropy(logits, label).mean(dim=-1) * mask
        losses = losses / torch.sum(mask).item()
        loss = losses.sum()
        loss_list.append(loss.item())

        if optimizers is not None:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

        acc_cnt += calc_accuracy(logits, label, mask)
        tot_cnt += torch.sum(mask).item()
    return sum(loss_list) / len(loss_list), acc_cnt / tot_cnt


def train(args, model, dataloaders, device):
    trainloader, valloader, testloader = dataloaders
    optimizers, schedulers = get_optimizers(args, model)

    for epoch in range(args.epochs):
        print(f'epoch: {epoch + 1}')
        model.train()
        train_avg_loss, train_avg_acc = step(trainloader, model, device, optimizers)

        model.eval()
        val_avg_loss, val_avg_acc = step(valloader, model, device, mode='val')
        test_avg_loss, test_avg_acc = step(testloader, model, device, mode='test')

        print(f'training,\t avg_loss={train_avg_loss},\t avg_acc={round(train_avg_acc * 100, 3)}')
        print(f'validating,\t avg_loss={val_avg_loss},\t avg_acc={round(val_avg_acc * 100, 3)}')
        print(f'testing,\t avg_loss={test_avg_loss},\t avg_acc={round(test_avg_acc * 100, 3)}')

        for scheduler in schedulers:
            scheduler.step()
