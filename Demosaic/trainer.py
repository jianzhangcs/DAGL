import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils import batch_PSNR
# loss_fun=torch.nn.MSE

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        #self.loss = torch.nn.MSELoss(size_average = False)#my_loss
        self.loss = torch.nn.L1Loss(size_average = False)
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        # self.loss.step()    # edit for new
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        # self.loss.start_log() # edit for new
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            #noise = torch.FloatTensor(hr.size()).normal_(mean=0, std=25. / 255.).cuda()
            #lr = hr + noise


            sr = self.model(lr, 1)#self.args.scale)
            # print(sr)
            # print(sr.shape)
            # exit(0)
            # print(hr.size()[0]*2)
            # exit(0)
            loss = self.loss(sr, hr)/(hr.size()[0]*2)
            # print(loss,(hr.size()[0]*2))
            psnr_train = batch_PSNR(sr, hr, 1.)
            psnr_org = batch_PSNR(lr,hr,1.)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()#data[0]
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.4f}\t{:.4f}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss.item(),#display_loss(batch),   # edit for new
                    psnr_train,
                    psnr_org,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        #if epoch%self.args.save_inter == 0 or epoch == self.args.epochs:
            #self.model.save(self.args.save_path,epoch=epoch,is_best=False)
            #torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'net_%s.pth' % epoch))
            # torch.save(model.state_dict(), os.path.join(self.args.save_path, 'net_%s.pth' % epoch))

        # self.loss.end_log(len(self.loader_train))   # edit for new
        # self.error_last = self.loss.log[-1, -1]     # edit for new

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        for idx_scale, scale in enumerate(self.scale):
            eval_acc = 0
            self.loader_test.dataset.set_scale(idx_scale)
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                # print(lr.shape,hr.shape)
                # exit(0)
                filename = filename[0]
                no_eval = isinstance(hr[0], int)
                if no_eval:
                    lr = self.prepare([lr], volatile=True)[0]
                else:
                    lr, hr = self.prepare([lr, hr], volatile=True)
                lr = lr#/255.
                hr = hr#/255.
                sr = torch.clamp(self.model(lr, idx_scale),0.,self.args.rgb_range)
                #sr = utility.quantize(sr, self.args.rgb_range)
                # psnr_test = batch_PSNR(sr, hr, 1.)

                save_list = [sr]
                if not no_eval:
                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    save_list.extend([lr, hr])

                if self.args.save_results:
                    if self.args.test_only:
                        self.ckp.save_results_test(filename, save_list, scale)
                    else:
                        self.ckp.save_results(filename, save_list, scale)

            self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} from epoch {})'.format(
                    self.args.data_test, scale, self.ckp.log[-1, idx_scale],
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        # print(self.args.test_only)
        # exit(0)
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[0][idx_scale] <= self.ckp.log[-1,idx_scale]))
            # print(best[1][0] + 1,epoch)
            #self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        def _prepare(idx, tensor):
            if not self.args.cpu: tensor = tensor.cuda()
            if self.args.precision == 'half': tensor = tensor.half()
            # Only test lr can be volatile
            return Variable(tensor, volatile=(volatile and idx==0))
           
        return [_prepare(i, _l) for i, _l in enumerate(l)]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

