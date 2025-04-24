# -*- coding: utf-8 -*-
# @Time    : 2021/8
# @Author  : Ruihong Qiu

import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
import datetime

from utils import recall_at_k, ndcg_k, get_metric

import nni
import pandas as pd


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args, sparsity_item_set):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device(f"cuda:{args.cuda}")

        self.model = model
        if self.cuda_condition:
            self.model.to(self.device)

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr,
                          betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement()
              for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

        self.sparsity_item_set = sparsity_item_set

    def train(self, epoch, ft_flag=None):
        self.iteration(epoch, self.train_dataloader, ft_flag=ft_flag)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        analyze_df = pd.DataFrame()
        analyze_df['answers'] = answers.reshape(1, -1)[0]
        for k in [5, 10, 15, 20]:
            recall_col_name = 'recall_' + str(k)
            ndcg_col_name = 'ndcg_' + str(k)
            recall_score, each_recall_list = recall_at_k(answers, pred_list, k)
            ndcg_score, each_ndcg_list = ndcg_k(answers, pred_list, k)

            analyze_df[recall_col_name] = each_recall_list
            analyze_df[ndcg_col_name] = each_ndcg_list
            recall.append(recall_score)
            ndcg.append(ndcg_score)
            
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        df_path = self.args.log_dir + '/'  + 'analyze_df.csv'
        if epoch == 999:
            nni.report_final_result(recall[0])
            analyze_df.to_csv(df_path)
        else:
            nni.report_intermediate_result(recall[0])
        # with open(self.args.log_file, 'a') as f:
        with open(self.args.log_dir + '/log.txt', 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def bpr(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        # [batch*seq_len hidden_size]
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) *
                                      self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def cross_entropy(self, seq_out, pos_ids, seq_len):
        indi_out = torch.cat([indi_seq[:indi_len]
                             for indi_seq, indi_len in zip(seq_out, seq_len)])
        target = torch.cat([indi_tar[:indi_len]
                           for indi_tar, indi_len in zip(pos_ids, seq_len)])
        pred = torch.mm(indi_out, self.model.item_embeddings.weight[1:].t())
        return self.model.ce(pred, target - 1)

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(
            test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def predict_full_att(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.all_fused_embedding
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class MMInfoRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args, sparsity_item_set):
        super(MMInfoRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args, sparsity_item_set
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True, ft_flag=None):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (
                                      str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            start_time = datetime.datetime.now()
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                # milnce
                loss = self.model.finetune(
                    batch, 'train', self.sparsity_item_set, ft_flag)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                'start_time': start_time, 
                'finish_time': datetime.datetime.now(), 
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            # with open(self.args.log_file, 'a') as f:
            with open(self.args.log_dir + '/log.txt', 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()
            start_time = datetime.datetime.now()

            pred_list = None

            answer_list = None
            seq_emb_list = []
            with torch.no_grad():
                self.model.cal_test_emb()
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, answer, _, seq_len, _, _ = batch
                    recommend_output = self.model.finetune(batch, flag='test')
                    seq_emb_list.append(torch.cat((user_ids.reshape(-1, 1), recommend_output), dim=1).cpu().numpy())
                    rating_pred = self.predict_full_att(recommend_output)

                    answers = answer.unsqueeze(dim=1)
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[torch.tensor(
                        self.args.train_matrix[batch_user_index].toarray() > 0)] = 0

                    _, batch_pred_list = rating_pred.topk(20)

                    if i == 0:
                        pred_list = batch_pred_list.cpu().numpy()
                        answer_list = answers.cpu().numpy()
                    else:
                        pred_list = np.append(
                            pred_list, batch_pred_list.cpu().numpy(), axis=0)
                        answer_list = np.append(
                            answer_list, answers.cpu().numpy(), axis=0)
                post_fix = {
                'start_time': start_time, 
                'finish_time': datetime.datetime.now(), 
                }

                with open(self.args.log_dir + '/log.txt', 'a') as f:
                    f.write(str(post_fix) + '\n')
                np.savetxt(f"{self.args.log_dir}/seq_embs.csv", 
                           np.concatenate(seq_emb_list, axis=0), 
                           delimiter=",")
            
            return self.get_full_sort_score(epoch, answer_list, pred_list)
