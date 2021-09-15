import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
from torch import nn

device_ids = [0]

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
torch.cuda.set_device(0)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
#     model = nn.DataParallel(module=model, device_ids=device_ids)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    if os.path.isdir(args.save_path) == False:
        os.mkdir(args.save_path)
        os.mkdir(os.path.join(args.save_path,'model'))
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()

