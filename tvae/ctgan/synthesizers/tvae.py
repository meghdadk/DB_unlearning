"""TVAESynthesizer module."""

import numpy as np
import wandb
from itertools import cycle
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer


class Encoder(Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAESynthesizer.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:

            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


def _kd_loss(teacher_logits, student_logits, sigmas=None, output_info=None, factor=None, _type="MSE"):
    if _type=="MSE":
        criteria = torch.nn.MSELoss()
        _loss = criteria(teacher_logits, student_logits)

    elif _type=="cross_entropy":
        assert sigmas is not None
        assert output_info is not None

        st = 0
        loss = []
        for column_info in output_info:
            for span_info in column_info:

                if span_info.activation_fn != 'softmax':
                    ed = st + span_info.dim
                    std = sigmas[st]
                    eq = torch.tanh(teacher_logits[:, st]) - torch.tanh(student_logits[:, st])
                    loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                    loss.append(torch.log(std) * teacher_logits.size()[0])
                    st = ed

                else:
                    ed = st + span_info.dim
                    loss.append(cross_entropy(
                        student_logits[:, st:ed], torch.argmax(teacher_logits[:, st:ed], dim=-1), reduction='sum'))
                    st = ed

        assert st == student_logits.size()[1]
        _loss = sum(loss) * factor / teacher_logits.size()[0]



    return _loss


class TVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        self.data_dim = 0

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

    def fit(self, train_data, discrete_columns=(), epochs=None, retrain=False, logger=None):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if ~retrain:
            self.transformer = DataTransformer()
            self.transformer.fit(train_data, discrete_columns)
            self.data_dim = self.transformer.output_dimensions

        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        
        encoder = Encoder(self.data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, self.data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)


        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.watch(self.decoder)

        if epochs is not None:
            self.epochs = epochs
        for i in range(self.epochs):
            ep_loss = 0
            ep_loss1 = 0
            ep_loss2 = 0
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                ep_loss += loss
                ep_loss1 += loss_1
                ep_loss2 += loss_2
                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tloss1: {:0.4f}\tloss2 : {:0.4f}\tloss : {:0.4f}".format(i,ep_loss1/len(loader), ep_loss2/len(loader),ep_loss/len(loader)))
            if logger is not None:
                logger.log({"loss1": ep_loss1/len(loader), "loss2": ep_loss2/len(loader), "loss": ep_loss/len(loader)})

        self.encoder = encoder    


    def finetune(self, train_data, discrete_columns=(),epochs=None, logger=None):
        if epochs is not None:
            self.epochs = epochs

        train_data = self.transformer.transform(train_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)


        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.watch(self.decoder)

        for i in range(self.epochs):
            ep_loss = 0
            ep_loss1 = 0
            ep_loss2 = 0
            j = 0
            for idx, data in enumerate(loader):
                j += 1
                optimizerAE.zero_grad()
                real = data[0].to(self._device)


                #training block
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2

                loss.backward()
                optimizerAE.step()
                ep_loss += loss
                ep_loss1 += loss_1
                ep_loss2 += loss_2


                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tloss1: {:0.4f}\tloss2 : {:0.4f}\tloss : {:0.4f}".format(i,ep_loss1/j, ep_loss2/j,ep_loss/j))
            if logger is not None:
                logger.log({"loss1": ep_loss1/len(loader), "loss2": ep_loss2/len(loader), "loss": ep_loss/len(loader)})

    def neggrad(self, train_data, deleted_data, discrete_columns=(),epochs=None, alpha=0.9, logger=None, args=None):
        if epochs is not None:
            self.epochs = epochs

        train_data = self.transformer.transform(train_data)
        deleted_data = self.transformer.transform(deleted_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        dataset_deleted = TensorDataset(torch.from_numpy(deleted_data.astype('float32')).to(self._device))
        loader_deleted = DataLoader(dataset_deleted, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.watch(self.decoder)
            
        for i in range(self.epochs):
            ep_loss = 0
            ep_loss1 = 0
            ep_loss2 = 0
            j = 0
            for idx, (data, data_del) in enumerate(zip(loader, cycle(loader_deleted))):
                j += 1
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                real_del = data_del[0].to(self._device)


                #neggrad block
                mu, std, logvar = self.encoder(real_del)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real_del, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss_del = loss_1 + loss_2

                #finetune block
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss_reg = loss_1 + loss_2


                loss = alpha*loss_reg - (1-alpha)*loss_del

                loss.backward()
                optimizerAE.step()
                ep_loss += loss
                ep_loss1 += loss_1
                ep_loss2 += loss_2


                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tloss1: {:0.4f}\tloss2 : {:0.4f}\tloss : {:0.4f}".format(i,ep_loss1/j, ep_loss2/j,ep_loss/j))
            if logger is not None:
                logger.log({"loss1": ep_loss1/len(loader), "loss2": ep_loss2/len(loader), "loss": ep_loss/len(loader)})

    def sgda(self, train_data, deleted_data, alpha, discrete_columns=(),epochs=None, logger=None):
        if epochs is not None:
            self.epochs = epochs

        train_data = self.transformer.transform(train_data)
        deleted_data = self.transformer.transform(deleted_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        dataset_deleted = TensorDataset(torch.from_numpy(deleted_data.astype('float32')).to(self._device))
        loader_deleted = DataLoader(dataset_deleted, batch_size=self.batch_size, shuffle=True, drop_last=False)


        t_decoder = self.decoder


        encoder = Encoder(self.data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, self.data_dim).to(self._device)
        optimizerAE_decoder = Adam(
            list(self.decoder.parameters()),
            weight_decay=self.l2scale)


        optimizerAE_encoder = Adam(
            list(encoder.parameters()),
            weight_decay=self.l2scale)

        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.watch(self.decoder)
            
        t_decoder.eval()
        for i in range(self.epochs):
            enc_loss = 0
            dec_loss = 0
            for idx, (data, data_del) in enumerate(zip(loader, cycle(loader_deleted))):
                optimizerAE_encoder.zero_grad()
                optimizerAE_decoder.zero_grad()
                real = data[0].to(self._device)
                real_del = data_del[0].to(self._device)


                #distillation block
                mean = torch.zeros(self.batch_size, self.embedding_dim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self._device)

                t_fake, t_sigmas = t_decoder(noise)
                s_fake, s_sigmas = self.decoder(noise)


                kd_loss = _kd_loss(t_fake, s_fake)
                 #end distillation block

                #training block
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                train_loss = loss_1 + loss_2

                loss_encoder = train_loss

                loss_decoder = alpha*train_loss + (1-alpha)*kd_loss

                loss_encoder.backward(retain_graph=True)
                loss_decoder.backward()
                optimizerAE_decoder.step()
                optimizerAE_encoder.step()


                enc_loss += loss_encoder
                dec_loss += loss_decoder


                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tencoder loss: {:0.4f}\tdecoder loss : {:0.4f}".format(i,enc_loss/len(loader), dec_loss/len(loader)))

        self.encoder = encoder

    def update(self, transfer_data, train_data, alpha, discrete_columns=(),epochs=None):
        if epochs is not None:
            self.epochs = epochs

        train_data = self.transformer.transform(train_data)
        transfer_data = self.transformer.transform(transfer_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)


        tr_dataset = TensorDataset(torch.from_numpy(transfer_data.astype('float32')).to(self._device))
        tr_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)


        pre_encoder = self.encoder
        pre_decoder = self.decoder


        encoder = Encoder(self.data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, self.data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)


        pre_decoder.eval()
        pre_decoder.eval()
        for i in range(self.epochs):
            ep_loss = 0
            ep_tloss = 0
            ep_kdloss = 0
            j = 0
            for data, tr_data in zip(loader,cycle(tr_loader)):
            #for idx, data in enumerate(loader):
                j += 1
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                real_tr = tr_data[0].to(self._device)


                #distillation block
                pre_mu, pre_std, pre_logvar = pre_encoder(real_tr)
                pre_eps = torch.randn_like(pre_std)
                pre_emb = pre_eps * pre_std + pre_mu
                pre_rec, pre_sigmas = pre_decoder(pre_emb)

                mu, std, logvar = encoder(real_tr)
                #eps = torch.randn_like(std)
                emb = pre_eps * std + mu
                rec, sigmas = self.decoder(emb)


                loss_1_kd = _kd_loss(pre_logvar, logvar)
                loss_2_kd = _kd_loss(pre_rec, rec)

                kd_loss = loss_1_kd + loss_2_kd
                 #end distillation block

                #training block
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                train_loss = loss_1 + loss_2


                loss = alpha*train_loss + (1-alpha)*kd_loss

                loss.backward()
                optimizerAE.step()
                ep_loss += loss
                ep_tloss += train_loss
                ep_kdloss += kd_loss


                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tloss1: {:0.4f}\tloss2 : {:0.4f}\tloss : {:0.4f}".format(i,ep_tloss/j, ep_kdloss/j,ep_loss/j))

        self.encoder = encoder

    def update_decoder(self, train_data, alpha, discrete_columns=(),epochs=None):
        if epochs is not None:
            self.epochs = epochs

        train_data = self.transformer.transform(train_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)


        t_decoder = self.decoder


        encoder = Encoder(self.data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, self.data_dim).to(self._device)
        optimizerAE_decoder = Adam(
            list(self.decoder.parameters()),
            weight_decay=self.l2scale)


        optimizerAE_encoder = Adam(
            list(encoder.parameters()),
            weight_decay=self.l2scale)


        t_decoder.eval()
        for i in range(self.epochs):
            enc_loss = 0
            dec_loss = 0
            for idx, data in enumerate(loader):
                optimizerAE_encoder.zero_grad()
                optimizerAE_decoder.zero_grad()
                real = data[0].to(self._device)


                #distillation block
                mean = torch.zeros(self.batch_size, self.embedding_dim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self._device)

                t_fake, t_sigmas = t_decoder(noise)
                s_fake, s_sigmas = self.decoder(noise)


                kd_loss = _kd_loss(t_fake, s_fake)
                 #end distillation block

                #training block
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                train_loss = loss_1 + loss_2

                loss_encoder = train_loss

                loss_decoder = alpha*train_loss + (1-alpha)*kd_loss

                loss_encoder.backward(retain_graph=True)
                loss_decoder.backward()
                optimizerAE_decoder.step()
                optimizerAE_encoder.step()


                enc_loss += loss_encoder
                dec_loss += loss_decoder


                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tencoder loss: {:0.4f}\tdecoder loss : {:0.4f}".format(i,enc_loss/len(loader), dec_loss/len(loader)))

        self.encoder = encoder

    def distill(self, train_data, discrete_columns=(),epochs=None):
        if epochs is not None:
            self.epochs = epochs

        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)


        pre_encoder = self.encoder
        pre_decoder = self.decoder


        encoder = Encoder(self.data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, self.data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)


        pre_decoder.eval()
        pre_decoder.eval()
        for i in range(self.epochs):
            ep_loss = 0
            ep_loss1 = 0
            ep_loss2 = 0
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)

                pre_mu, pre_std, pre_logvar = pre_encoder(real)
                pre_eps = torch.randn_like(pre_std)
                pre_emb = pre_eps * pre_std + pre_mu
                pre_rec, pre_sigmas = pre_decoder(pre_emb)

                mu, std, logvar = encoder(real)
                #eps = torch.randn_like(std)
                emb = pre_eps * std + mu
                rec, sigmas = self.decoder(emb)


                loss_1 = _kd_loss(pre_logvar, logvar)
                loss_2 = _kd_loss(pre_rec, rec)

                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                ep_loss += loss
                ep_loss1 += loss_1
                ep_loss2 += loss_2
                self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tloss1: {:0.4f}\tloss2 : {:0.4f}\tloss : {:0.4f}".format(i,ep_loss1/len(loader), ep_loss2/len(loader),ep_loss/len(loader)))

        self.encoder = encoder

    def distill_decoder(self,epochs=None):
        if epochs is not None:
            self.epochs = epochs


        t_decoder = self.decoder

        self.decoder = Decoder(self.embedding_dim, self.compress_dims, self.data_dim).to(self._device)
        optimizerAE = Adam(list(self.decoder.parameters()),weight_decay=self.l2scale)


        t_decoder.eval()
        for i in range(self.epochs):
            optimizerAE.zero_grad()

            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)

            t_fake, t_sigmas = t_decoder(noise)

            s_fake, s_sigmas = self.decoder(noise)


            loss_1 = _kd_loss(t_fake, s_fake,s_sigmas, self.transformer.output_info_list, self.loss_factor, _type="cross_entropy")
            #loss_1 = _kd_loss(t_fake, s_fake, _type='MSE')
            loss = loss_1 
            loss.backward()
            optimizerAE.step()

            self.decoder.sigma.data.clamp_(0.01, 1.0)


            print ("epoch {}\tloss : {}".format(i,loss))

    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
