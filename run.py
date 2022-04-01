# coding: UTF-8
import time
import torch
import numpy as np

from models.FGM import FGM
from train_eval import train, init_network
from importlib import import_module
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--adversarial', type=str, required=False, help='choose an adversarial method: fgm, pgd, freeAT')
args = parser.parse_args()

def fgm_train(model):
    fgm = FGM(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for i, (batch_input, batch_label) in enumerate(train_iter):
        # 正常训练
        outputs = model(batch_input)
        model.zero_grad()
        loss = F.cross_entropy(outputs, batch_label)
        print("loss")
        print(loss)
        loss.backward()  # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        outputs = model(batch_input)
        model.zero_grad()
        loss_adv = F.cross_entropy(outputs, batch_label)
        print("loss_adv")
        print(loss_adv)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif, flush=True)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters, flush=True)
    # print("=================use fgm===============", flush=True)
    adversarial = args.adversarial
    with open("./" + str(int(time.time())) + ".txt", mode="w") as result_file:
        print(adversarial)
        train(config, model, train_iter, dev_iter, test_iter, adversarial, result_file)
        # train(config, model, train_iter, dev_iter, test_iter, "pgd", result_file)
        # train(config, model, train_iter, dev_iter, test_iter, "fgm", result_file)
        # train(config, model, train_iter, dev_iter, test_iter, None, result_file)
