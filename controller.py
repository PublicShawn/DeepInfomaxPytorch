from model.models import Encoder
from loss.deepinfomax import DeepInfoMaxLoss
from model.latentdim import LatentDIM
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import torch
from dataset.DataLoadingManager import DataLoadingManager
import statistics as stats
from utils.funcs import precision, compute_result, CalcTopMap, one_hot_embedding

class Controller(object):
    def __init__(self, config):
        self.config = config.get()
        if self.config.dataset is None:
            raise FileNotFoundError("missing dataset")
        self.datasetmanager = DataLoadingManager(self.config.dataset, self.config.dataroot, self.config.batch_size, 0.2)
        self.dataset = self.datasetmanager()

    def build_train(self):
        self.encoder = Encoder(self.config.hashbit).to(self.config.device)
        self.loss_fn = DeepInfoMaxLoss(self.config.hashbit).to(self.config.device)
        self.quant_lossfn = torch.nn.MSELoss()
        self.optim = Adam(self.encoder.parameters(), lr=1e-4)
        self.loss_optim = Adam(self.loss_fn.parameters(), lr=1e-4)

        epoch_restart = self.config.epochrestart
        modelroot = None if not self.config.modelroot else self.config.modelroot

        if epoch_restart is not None and modelroot is not None:
            enc_file = modelroot / Path('encoder' + str(epoch_restart) + '.wgt')
            loss_file = modelroot / Path('loss' + str(epoch_restart) + '.wgt')
            self.encoder.load_state_dict(torch.load(str(enc_file)))
            self.loss_fn.load_state_dict(torch.load(str(loss_file)))

    def build_test(self):
        self.classifier = LatentDIM(self.encoder).to(self.config.device)
        self.clsoptim = Adam(self.classifier.parameters(), lr=1e-4)
        self.clscriterion = torch.nn.CrossEntropyLoss()


    def train(self):
        print("start training......")
        batch_size = self.config.batch_size
        epoch_restart = 0 if self.config.epochrestart is None else self.config.epochrestart
        modelroot = None if not self.config.modelroot else self.config.modelroot
        for epoch in range(epoch_restart + 1, self.config.epochs):
            batch = tqdm(self.dataset.train, total=self.dataset.train_sz // batch_size)
            train_loss = []
            quant_train_loss = []

            for x, target in batch:
                x = x.to(self.config.device)

                self.optim.zero_grad()
                self.loss_optim.zero_grad()
                y, M = self.encoder(x)
                # rotate images to create pairs for comparison
                M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                loss = self.loss_fn(y, M, M_prime)
                quant_loss = self.quant_lossfn(y, torch.sign(y).detach())
                train_loss.append(loss.item())
                quant_train_loss.append(quant_loss.item())
                batch.set_description(str(epoch) + 'DIM Loss: ' + str(stats.mean(train_loss[-20:])) + " Q Loss" + str(stats.mean(quant_train_loss[-20:])))
                quant_loss.backward(retain_graph=True)
                loss.backward()

                self.optim.step()
                self.loss_optim.step()
            mAp = self.evalmap()
            print("epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (
                epoch + 1, self.config.hashbit, self.config.dataset, mAp))
            print()

            if epoch % self.config.saveepoch == 0:
                modelrootpath = Path(modelroot)
                enc_file = modelrootpath / Path('encoder' + str(epoch) + '.wgt')
                loss_file = modelrootpath / Path('loss' + str(epoch) + '.wgt')
                enc_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.encoder.state_dict(), str(enc_file))
                torch.save(self.loss_fn.state_dict(), str(loss_file))

    def evalmap(self):
        # print("calculating test binary code......")
        tst_binary, tst_label = compute_result(self.dataset.test, self.encoder, device=self.config.device)

        # print("calculating dataset binary code.......")\
        trn_binary, trn_label = compute_result(self.dataset.train, self.encoder, device=self.config.device)

        # print("calculating map.......")
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), one_hot_embedding(trn_label.numpy(), self.dataset.num_classes), one_hot_embedding(tst_label.numpy(), self.dataset.num_classes),
                         self.config.topK)
        return mAP


    def test(self):
        print("start testing......")
        batch_size = self.config.batch_size
        for epoch in range(self.config.epochs):

            ll = []
            batch = tqdm(self.dataset.train, total=self.dataset.train_sz // batch_size)
            for x, target in batch:

                x = x.to(self.config.device)
                target = target.to(self.config.device)

                self.clsoptim.zero_grad()
                y = self.classifier(x)
                loss = self.clscriterion(y, target)
                ll.append(loss.detach().item())
                batch.set_description(f'{epoch} Train Loss: {stats.mean(ll)}')
                loss.backward()
                self.clsoptim.step()

            confusion = torch.zeros(self.dataset.num_classes, self.dataset.num_classes)
            batch = tqdm(self.dataset.test, total=self.dataset.test_sz // batch_size )
            ll = []
            for x, target in batch:
                x = x.to(self.config.device)
                target = target.to(self.config.device)

                y = self.classifier(x)
                loss = self.clscriterion(y, target)
                ll.append(loss.detach().item())
                batch.set_description(f'{epoch} Test Loss: {stats.mean(ll)}')

                _, predicted = y.detach().max(1)

                for item in zip(predicted, target):
                    confusion[item[0], item[1]] += 1

            precis = precision(confusion)
            print(precis)
