

import listwise_ltr
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import time
import sys
import os
import pickle
sys.path.append('..')


def ndcg(y_hat, y):

    def dcg(y_hat, y):
        sort_indices = y_hat.sort(descending=True)[1]

        return ((2**y[sort_indices].float()-1) / torch.arange(2, len(y)+2).to(device).float().log()).sum()
    norm = dcg(y, y)
    if norm == 0:
        return torch.tensor(1)
    else:
        return dcg(y_hat, y) / norm


def err(y_hat, y):
    R = y[y_hat.sort(descending=True).indices]
    Nd = len(R)

    phi = torch.arange(1, Nd+1).to(device)
    p = (2**R-1) / 2**4
    prob_stopping = torch.cumprod(1-p, 0) / (1-p[0]) * p
    return (prob_stopping/phi).sum()


def delta_ndcg(y_hat, y):

    indices = y_hat.sort(descending=True).indices
    R_true = y.sort(descending=True).values
    R = y[indices]

    alpha = ((2 ** R_true - 1) / torch.arange(2,
                                              len(y)+2).float().log2().to(device)).sum()

    if alpha.item() == 0:
        return torch.ones_like(R)
    else:
        Nd = len(y_hat)

        Delta = ((2**R-1).unsqueeze(1) / torch.arange(2,
                                                      len(R)+2).float().log2().to(device))
        ddcg = (Delta.sum(dim=1) + Delta.sum(dim=0) -
                Delta.diag() * Nd - Delta.diag().sum())/alpha

        return ddcg[indices] / alpha


def delta_err(y_hat, y):
    R = y[y_hat.sort(descending=True).indices]
    Nd = len(R)
    D = torch.zeros(Nd, Nd, Nd).to(device)

    phi = torch.arange(1, Nd+1).to(device)
    p = (2**R-1) / 2**4

    for i in range(Nd):
        for j in range(i+1, Nd):

            # swap indices
            p = p.clone()
            p[i], p[j] = p[j].item(), p[i].item()

            prob_stopping = torch.cumprod(1-p, 0) / (1-p[0]) * p
            D[i, j, :] = D[j, i, :] = prob_stopping

    D = (D / phi).sum(dim=[1, 2])

    return D


class LambdaRankFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y_hat, y, sigma, metric):

        S = (y.unsqueeze(1) - y.t()).clamp(-1, 1)
        D = y_hat.unsqueeze(1) - y_hat.t()

        ctx.save_for_backward(S, D, y_hat, y)
        ctx.sigma = sigma
        ctx.metric = metric

        loss = .5 * (1-S) * sigma * D + torch.log2(1 + torch.exp(-sigma*D))

        return loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        S, D, y_hat, y = ctx.saved_tensors

        Lambda = ctx.sigma * (.5 * (1-S) - 1 / (1+torch.exp(-ctx.sigma*D)))

        if ctx.metric == 'NDCG':
            dirm = delta_ndcg(y_hat, y)
        elif ctx.metric == 'ERR':
            dirm = delta_err(y_hat, y)
        else:
            raise NotImplementedError()

        dy_hat = Lambda.sum(dim=1) * dirm.abs()

        return dy_hat, None, None, None


class LambdaRankLoss(torch.nn.Module):
    '''
    input: predictions and target
    output: scalar loss 
    '''

    def __init__(self, sigma, metric):
        assert metric in ['NDCG', 'ERR']

        super(LambdaRankLoss, self).__init__()
        self.sigma = sigma
        self.metric = metric

    def forward(self, y_hat, y):
        return LambdaRankFunction.apply(y_hat, y, self.sigma, self.metric)


def train(train_dl, valid_dl, config, max_epochs, early_stopping_metric, patience, verbose=True, plot=False, name=None):
    train_loss = defaultdict(list)
    valid_loss = defaultdict(list)
    train_ndcg = defaultdict(list)
    valid_ndcg = defaultdict(list)
    train_err = defaultdict(list)
    valid_err = defaultdict(list)

    model = listwise_ltr.Listwise()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    critereon = LambdaRankLoss(sigma=config['sigma'], metric=config['metric'])

    tqdm_str_train = 'TRAIN: loss: {0:.3f}\t NDCG: {1:.3f}\t ERR: {2:.3f}'
    tqdm_str_valid = 'TRAIN: loss: {0:.3f}\t NDCG: {1:.3f}\t ERR: {2:.3f} \t VALID: loss: {3:.3f}\t NDCG: {4:.3f} \t ERR: {5:.3f}'

    for epoch in range(max_epochs):

        # iterate over training set
        if verbose:
            pb = tqdm(total=len(train_dl))
            pb.set_description(f'Epoch: {epoch+1}/{max_epochs}')

        model.train()
        for ix, (X, y) in enumerate(train_dl):
            X = X.to(device).float().squeeze(0)
            y = y.to(device).float().squeeze(0)

            y_hat = model(X).sigmoid().squeeze(1)

            # compute loss
            loss = critereon(y_hat, y)

            optimizer.zero_grad()
            y_hat.sum().backward(loss)
            # print(model.nnet[2].weight)
            optimizer.step()

            train_loss[epoch].append(loss.item())
            ndcg_batch = ndcg(y_hat, y).item()
            err_batch = err(y_hat, y).item()

            train_ndcg[epoch].append(ndcg_batch)
            train_err[epoch].append(err_batch)

            if ix % 50 == 0 and verbose:
                pb.set_postfix_str(tqdm_str_train.format(
                    loss.item(), ndcg_batch, err_batch
                ))

            if verbose:
                pb.update()

        mean_train_ndcg = np.mean(train_ndcg[epoch])
        mean_train_err = np.mean(train_err[epoch])
        mean_train_loss = np.mean(train_loss[epoch])

        # validate
        with torch.no_grad():
            model.eval()
            for ix, (X, y) in enumerate(valid_dl):
                X = X.to(device).float().squeeze(0)
                y = y.to(device).float().squeeze(0)

                y_hat = model(X).sigmoid().squeeze(1)

                # compute loss
                loss = critereon(y_hat, y)
                ndcg_batch = ndcg(y_hat, y).item()
                err_batch = err(y_hat, y).item()

                valid_loss[epoch].append(loss.item())
                valid_ndcg[epoch].append(ndcg_batch)
                valid_err[epoch].append(err_batch)

                if ix % 50 == 0 and verbose:
                    pb.set_postfix_str(tqdm_str_valid.format(
                        mean_train_loss, mean_train_ndcg, mean_train_err, loss.item(), ndcg_batch, err_batch
                    ))

        mean_valid_ndcg = np.mean(valid_ndcg[epoch])
        mean_valid_err = np.mean(valid_err[epoch])
        mean_valid_loss = np.mean(valid_loss[epoch])

        if verbose:
            pb.set_postfix_str(tqdm_str_valid.format(
                mean_train_loss, mean_train_ndcg, mean_train_err, mean_valid_loss, mean_valid_ndcg, mean_valid_err
            ))

    if plot:
        plot_loss_ndcg(train_loss, valid_loss, train_ndcg,
                       valid_ndcg, train_err, valid_err, name)

    return model, mean_valid_ndcg, mean_valid_err


def plot_loss_ndcg(train_loss, valid_loss, train_ndcg, valid_ndcg, train_err, valid_err, name):
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
    
    ax1.plot([np.mean(train_loss[epoch]) for epoch in train_loss], label='train')
    ax1.plot([np.mean(valid_loss[epoch]) for epoch in valid_loss], label='valid')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot([np.mean(train_ndcg[epoch]) for epoch in train_ndcg], label='train')
    ax2.plot([np.mean(valid_ndcg[epoch]) for epoch in valid_ndcg], label='valid')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NDCG')

    ax3.plot([np.mean(train_err[epoch]) for epoch in train_err], label='train')
    ax3.plot([np.mean(valid_err[epoch]) for epoch in valid_err], label='valid')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ERR')
    
    ax1.grid(linewidth=0.5, linestyle='--')
    ax1.legend()
    
    ax2.grid(linewidth=0.5, linestyle='--')
    ax2.legend()
    
    ax3.grid(linewidth=0.5, linestyle='--')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'figures/plot_progress_{name}.pdf', type='pdf')


def hyper_param_search(train_dl, valid_dl):
    #hyper param search

    # learning_rates = [0.001, 0.005, 0.01, 0.05]
    learning_rates = [0.001]
    # sigmas = [0, 0.25, .5, 1.]
    sigmas = [1.]
    metrics = ['NDCG']#, 'ERR']

    configs = [
        {'learning_rate':l, 'sigma':s, 'metric':m} for l in learning_rates for s in sigmas for m in metrics
    ]
    
    best_ndcg_config = None
    best_ndcg_model = None
    best_ndcg_score = -np.inf
    best_err_config = None
    best_err_model = None
    best_err_score = -np.inf
    
    # save best ndcg, best err
    for i, config in tqdm(enumerate(configs), total=len(configs)):
        model, eval_ndcg, eval_err = train(
            train_dl = train_dl,
            valid_dl = valid_dl, 
            config = config, 
            max_epochs=10,
            early_stopping_metric='NDCG',
            patience=2,
            verbose=False
        )

        if eval_ndcg > best_ndcg_score:
            best_ndcg_score = eval_ndcg
            best_ndcg_model = model
            best_ndcg_config = i

            print(f'[{time.ctime()}] Improved NDCG to {eval_ndcg} using {config}')

        if eval_err > best_err_score:
            best_err_score = eval_err
            best_err_model = model
            best_err_config = i

            print(f'[{time.ctime()}] Improved ERR to {eval_err} using {config}')


    return configs[best_ndcg_config], configs[best_err_config]


def early_stopping(valid_results, patience = 0, objective = 'max'):
    '''
    Returns True if valid_loss did not increase for the last 'patience' epochs.

    params:
        valid_results: dictionary with the validation metric results per epoch
        patience: number of allowed consecutive epochs without improvement
        mode: whether to maximize or minimize the metric

    '''
    
    if len(valid_results) <= 1:
        return False
    
    means = [np.mean(valid_results[epoch]) for epoch in valid_results]
    cur_epoch = len(means)
    diff = np.diff(means)    
   
    if objective == 'max':
        last_improvement = len(diff) - diff[::-1].argmax() + 1
    elif objective == 'min':
        last_improvement = len(diff) - diff[::-1].argmin() + 1
    else:
        raise NotImplementedError()
    
    stop = cur_epoch - last_improvement > patience

    return stop





def evaluate(model, config, dataloader, name):
    results = defaultdict(list)

    critereon = LambdaRankLoss(sigma=config['sigma'], metric=config['metric'])

    with torch.no_grad():
        model.eval()
        for ix, (X,y) in enumerate(dataloader):
            X = X.to(device).float().squeeze(0)
            y = y.to(device).float().squeeze(0)

            y_hat = model(X).sigmoid().squeeze(1)

            # compute metrics
            loss = critereon(y_hat, y)
            ndcg_batch = ndcg(y_hat, y).item()
            err_batch = err(y_hat, y).item()
            results['test_loss'].append(loss.item())
            results['test_ndcg'].append(ndcg_batch)
            results['test_err'].append(err_batch)
            

        mean_test_ndcg = np.mean(results['test_ndcg'])
        mean_test_loss = np.mean(results['test_loss'])
        mean_test_err  = np.mean(results['test_err'])
    
    with open(f'results/results_{name}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Loss: {np.mean(results['test_loss'])}, sd: {np.std(results['test_loss'])}")
    print(f"NDCG: {np.mean(results['test_ndcg'])}, sd: {np.std(results['test_ndcg'])}")
    print(f"ERR: {np.mean(results['test_err'])}, sd: {np.std(results['test_err'])}")

    return results


if __name__ == '__main__':
    import dataset
    device = 'cpu'
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    data= dataset.get_dataset()
    data.data_paths = ['../dataset']
    data = data.get_data_folds()[0]
    data.read_data()

    train_dataset = dataset.ListDataSet(data.train)
    valid_dataset = dataset.ListDataSet(data.validation)
    test_dataset = dataset.ListDataSet(data.test)

    train_dl = DataLoader(train_dataset, shuffle=True)
    valid_dl = DataLoader(valid_dataset)
    test_dl = DataLoader(test_dataset)


    # hyper param search
    print(f'[{time.ctime()}] Start hyper opt search')

    best_ndcg_config, best_err_config = hyper_param_search(train_dl, valid_dl)



    # train models for optimal configs for NDCG and ERR
    print(f'[{time.ctime()}] Training optimal NDCG model')
    best_ndcg_model, _, _ = train(
            train_dl = train_dl,
            valid_dl = valid_dl, 
            config = best_ndcg_config, 
            max_epochs=15,
            early_stopping_metric='NDCG',
            patience=1,
            verbose=False,
            plot=True,
            name='best_ndcg'
    )

    print(f'[{time.ctime()}] Training optimal ERR model')
    best_err_model, _, _ = train(
            train_dl = train_dl,
            valid_dl = valid_dl, 
            config = best_err_config, 
            max_epochs=15,
            early_stopping_metric='ERR',
            patience=1,
            verbose=False,
            plot=True,
            name='best_err'
    )

    # save models
    print(f'[{time.ctime()}] Saving models')
    torch.save(best_ndcg_model.state_dict(), 'models/best_ndcg_model')
    torch.save(best_err_model.state_dict(), 'models/best_err_model')


    # eval on test set
    print(f'[{time.ctime()}] Evaluating best models on test set')
    results_ndcg = evaluate(best_ndcg_model, best_ndcg_config, test_dl, name='best_ndcg')
    results_err = evaluate(best_err_model, best_err_config, test_dl, name='best_err')





