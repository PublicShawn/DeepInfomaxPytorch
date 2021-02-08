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
from nanopq.pq import PQ


class PQController(object):
    def __init__(self, config):
        self.config = config.get()
        if self.config.dataset is None:
            raise FileNotFoundError("missing dataset")
        self.datasetmanager = DataLoadingManager(self.config.dataset, self.config.dataroot, self.config.batch_size, 0.2)
        self.dataset = self.datasetmanager()

    def build_train(self):
        self.encoder = Encoder(self.config.hashbit).to(self.config.device)
        self.loss_fn = DeepInfoMaxLoss(self.config.hashbit).to(self.config.device)
        self.optim = Adam(self.encoder.parameters(), lr=1e-2)
        self.loss_optim = Adam(self.loss_fn.parameters(), lr=1e-4)

        epoch_restart = self.config.epochrestart
        modelroot = None if not self.config.modelroot else self.config.modelroot

        if epoch_restart is not None and modelroot is not None:
            enc_file = modelroot / Path('encoder' + str(epoch_restart) + '.wgt')
            loss_file = modelroot / Path('loss' + str(epoch_restart) + '.wgt')
            self.encoder.load_state_dict(torch.load(str(enc_file)))
            self.loss_fn.load_state_dict(torch.load(str(loss_file)))


    def train(self):
        print("start training......")
        batch_size = self.config.batch_size
        epoch_restart = 0 if self.config.epochrestart is None else self.config.epochrestart
        modelroot = None if not self.config.modelroot else self.config.modelroot

        for epoch in range(epoch_restart, self.config.epochs):
            batch = tqdm(self.dataset.train, total=self.dataset.train_sz // batch_size)
            train_loss = []
            cat = None

            for x, target in batch:
                x = x.to(self.config.device)
                if cat is None:
                    cat = target
                else:
                    cat = torch.cat([cat, target])
                self.optim.zero_grad()
                self.loss_optim.zero_grad()
                y, M = self.encoder(x)
                # rotate images to create pairs for comparison
                M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                loss = self.loss_fn(y, M, M_prime)
                train_loss.append(loss.item())
                batch.set_description(str(epoch) + 'DIM Loss: ' + str(stats.mean(train_loss[-20:])))
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

    def trainPQ(self):
        print("train pq")
        binary, _ = compute_result(self.dataset.train, self.encoder, device=self.config.device)
        binary = binary.detach().numpy()
        pq = PQ(M=8)
        pq.fit(binary)
        return pq

    def evalmap(self):
        pq = self.trainPQ()
        # print("calculating test binary code......")
        tst_binary, tst_label = compute_result(self.dataset.query, self.encoder, device=self.config.device)
        encoded_tst_binary = pq.encode(tst_binary.detach().numpy())
        decoded_tst_binary = pq.decode(encoded_tst_binary)
        tst_binary = torch.from_numpy(decoded_tst_binary).cpu()

        # print("calculating dataset binary code.......")\
        trn_binary, trn_label = compute_result(self.dataset.retrieve, self.encoder, device=self.config.device)
        encoded_trn_binary = pq.encode(trn_binary.detach().numpy())
        decoded_trn_binary = pq.decode(encoded_trn_binary)
        trn_binary = torch.from_numpy(decoded_trn_binary).cpu()

        # print("calculating map.......")
        # print(tst_binary[:17, :])
        # print("!!!!!!!!")
        # print(trn_binary[:17, :])
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), one_hot_embedding(trn_label.numpy(), self.dataset.num_classes), one_hot_embedding(tst_label.numpy(), self.dataset.num_classes),
                         trn_binary.shape[0])
        return mAP


