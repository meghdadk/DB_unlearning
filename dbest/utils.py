import numpy as np
import math
from scipy import integrate,stats


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

torch.manual_seed(1234)
np.random.seed(1234)

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step
    this is from https://github.com/HobbitLong/RepDistiller
    """
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.learning_rate
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def sample(pi, sigma, mu):
    """Draw samples from a MoG."""
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i, idx]).add(mu[i, idx])
    return sample

def MoGـsampling(pi, sigma, mu, n_samples, device):

    pis = pi

    indices = pis.multinomial(num_samples=n_samples, replacement=True)

    sigma = sigma.reshape(sigma.shape[0], sigma.shape[1])
    mu = mu.reshape(mu.shape[0], mu.shape[1])
    sigmas = torch.gather(sigma, 1, indices)
    mus = torch.gather(mu, 1, indices)
    samples = torch.normal(mus, sigmas)
   
    return samples

def gaussian_probability(sigma, mu, data):
    data = data.unsqueeze(1).expand_as(sigma)
    ret = (
        1.0
        / math.sqrt(2 * math.pi)
        * torch.exp(-0.5 * ((data - mu) / sigma) ** 2)
        / sigma
    )
    return torch.prod(ret, 2)

def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """

    data = data.unsqueeze(1).expand_as(sigma)
    ret = (
        1.0
        / math.sqrt(2 * math.pi)
        * torch.exp(-0.5 * ((data - mu) / sigma) ** 2)
        / sigma
    )
    return torch.prod(ret, 2)

def LL(pi, sigma, mu, target, device):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    softmax = nn.Softmax(dim=1)
    pi = softmax(pi)
    
    prob = pi * gaussian_probability(sigma, mu, target)

    ll = torch.log(torch.sum(prob, dim=1)).to(device)

    return torch.mean(ll)

def mdn_loss(pi, sigma, mu, target, device, reduction="mean", _return="nll"):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    softmax = nn.Softmax(dim=1)
    pi = softmax(pi)
    
    prob = pi * gaussian_probability(sigma, mu, target)

    if _return == "nll":
        loss = -torch.log(torch.sum(prob, dim=1)+1e-10).to(device)
    else:
        loss = torch.sum(prob, dim=1).to(device)
    
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "None":
        return loss
    else:
        raise ValueError("reduction type is not defined!")

    


def mse_kd_loss(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn, target, device, T=1):

    softmax = nn.Softmax(dim=1)
    pi_do = softmax(pi_do/T)
    pi_dn = softmax(pi_dn/1)
    prob = torch.sum(pi_do * gaussian_probability(sigma_do, mu_do, target),dim=1)
    prob_d = torch.sum(pi_dn * gaussian_probability(sigma_dn, mu_dn, target),dim=1)
    #prob_d = torch.log(torch.add(pi_dn * gaussian_probability(sigma_dn, mu_dn, target), 1e-7))
    #cross_entropy = -torch.sum(prob*prob_d,dim=1).to(device)
    #cross_entropy = -prob * torch.log(prob_d)
    #cross_entropy = cross_entropy.to(device)
    
    
    cross_entropy = torch.square(prob_d - prob).to(device)
    #cross_entropy = torch.log(torch.max(prob, prob_d)/torch.min(prob, prob_d))
    return torch.mean(cross_entropy)


def mse_ce_loss(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn, target, device, T=1): 

    KD = nn.KLDivLoss()(F.log_softmax(pi_dn/T, dim=1),F.softmax(pi_do/T, dim=1))

    mse_loss = nn.MSELoss(reduction="mean")
    mse_mus = mse_loss(mu_dn, mu_do)
    mse_sigmas = mse_loss(sigma_dn, sigma_do)


    kd_loss = KD + mse_mus + mse_sigmas
    return torch.mean(kd_loss)


def ks_kd_loss(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn):
    T = 1
    softmax = nn.Softmax(dim=1)
    pi_do = softmax(pi_do/T)
    pi_dn = softmax(pi_dn/T)

    samples1 = MoGsampling(pi_do, sigma_do, mu_do,1000,device)
    samples2 = MoGsampling(pi_dn, sigma_dn, mu_dn,1000,device)   

    stat = []
    for i, row in enumerate(samples1):
        s , p = stats.ks_2samp(row.detach().numpy(), samples2[i].detach().numpy())
        stat.append(s)

    loss = torch.log(Variable(torch.mean(torch.as_tensor(stat)), requires_grad=True)).to(device)
    return loss
    

def MoG_kl_divergance(MoG1, MoG2, device):
   
    pis1,sigmas1,mus1 = MoG1[0],MoG1[1],MoG1[2]
    pis2,sigmas2,mus2 = MoG2[0],MoG2[1],MoG2[2]    

    T = 1
    softmax = nn.Softmax(dim=1)
    pis1 = softmax(pis1/T)
    pis2 = softmax(pis2/T)

    samples1 = MoGsampling(pis1, sigmas1, mus1,10000,device)
    samples2 = MoGsampling(pis2, sigmas2, mus2,10000,device)

    samples1 = [i for i in samples1[0]]
    samples2 = [i for i in samples2[0]]

    kl = scipy.stats.entropy(samples1,samples2,base=2)
    return kl


def entropy_with_sampling(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn, device):

    T = 1
    softmax = nn.Softmax(dim=1)
    pi_do = softmax(pi_do/T)
    pi_dn = softmax(pi_dn/T)
    
    samples = MoGsampling(pi_do, sigma_do, mu_do, 100, device) 
    #samples_dn = MoGsampling(pi_dn, sigma_dn, mu_dn, 100, device)

    entropies = torch.zeros(samples.shape[0],1)
    for i in range(samples.shape[1]):
        probs_do = torch.sum(pi_do * gaussian_probability(sigma_do, mu_do, samples[:,i].unsqueeze(1)), dim=1) 
        probs_dn = torch.sum(pi_dn * gaussian_probability(sigma_dn, mu_dn, samples[:,i].unsqueeze(1)), dim=1) 

        entropy = -probs_do * torch.log(probs_dn/probs_do)
        #entropy = torch.log(torch.max(probs_dn, probs_do)/torch.min(probs_dn, probs_do))
        #entropy = torch.abs(probs_dn - probs_do)
        entropies = torch.cat([entropies, entropy.unsqueeze(1)],1)

    cross_entropy = torch.sum(entropies, dim=1).to(device)
    return torch.mean(cross_entropy) 




def gaussion_predict(weights: list, mus: list, sigmas: list, xs: list, n_jobs=1):
    if n_jobs == 1:
        result = np.array(
            [
                np.multiply(stats.norm(mus, sigmas).pdf(x), weights)
                .sum(axis=1)
                .tolist()
                for x in xs
            ]
        ).transpose()
    else:
        with Pool(processes=n_jobs) as pool:
            instances = []
            results = []
            for x in xs:
                i = pool.apply_async(gaussion_predict, (weights, mus, sigmas, [x], 1))
                instances.append(i)
            for i in instances:
                result = i.get()
                # print("partial result", result)
                results.append(result)
            result = np.concatenate(results, axis=1)

    return result


def denormalize(x_point, mean, width):
    """de-normalize the data point

    Args:
        x_point (float): the data point
        mean (float): the mean value
        width (float): the width

    Output:
        float: the de-normalized value
    """
    return 0.5 * width * x_point + mean


