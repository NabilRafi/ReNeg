# ---------------------------------------------------------------------------
# VENDORED OFFICIAL CODE - DO NOT EDIT (used only for equivalence tests).
# Source : https://github.com/YBZh/OpenOOD-VLM  (commit c6fef2f5ed890df2f59f9c6c3f9ae14fb5a72ab9)
# File   : openood/postprocessors/oneoodprompt_postprocessor.py (class OneOodPromptDevelopPostprocessor, lines 179-452)
# License: MIT (Copyright (c) 2021 Jingkang Yang; see OpenOOD-VLM/LICENSE).
# Changes: only the package-relative imports were replaced by local shims
#          (BasePostprocessor, openood.utils.comm). Algorithm code is verbatim.
# ---------------------------------------------------------------------------
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from oodlab.official._shims import BasePostprocessor, comm  # shim replacing repo imports
import pdb

class OneOodPromptDevelopPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OneOodPromptDevelopPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        ### get the image feature of each classes, construct the image feature classifier/memory.
        net.eval()
        # net.text_features 11k*512, if empty, fill with net.
        out_dim = net.n_output
        if self.setup_flag:
            # estimate class mean from training set
            # net.text_features.t() ## N*512
            # net.logit_scale
            # with torch.no_grad():
            #     output_text = net.logit_scale * net.text_features.t() @ net.text_features # class_num * class_num
            #     output_text = torch.softmax(output_text, dim=1) 
            # all_weights_text = output_text[:, :1000].sum(1) # prob of ID 
            # self.text_idscore_cache = all_weights_text ## 11k

            with torch.no_grad():
                output_text = net.logit_scale * net.text_features_unselected.t() @ net.text_features # class_num * class_num
                output_text = torch.softmax(output_text, dim=1) 
            all_weights_text = output_text[:, :1000].sum(1) # prob of ID 
            self.text_idscore_unselected_cache = all_weights_text ## 11k

            print('\n geting image features from (generated) training set...')
            all_feats = []
            all_weights = []
            for i in range(out_dim):
                all_feats.append([])  ## category-wise feature list
                all_weights.append([])
            
            ############################## init with text feature.
            # # pdb.set_trace()
            # for i in range(out_dim):
            #     all_feats[i].append(net.text_features.t()[i].unsqueeze(0))  ## category-wise feature list

            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    image_features, text_features, logit_scale = net(data, return_feat=True)
                    ####### weighting image features according to the classification probability.
                    output = logit_scale * image_features @ text_features.t() # batch * class, using the classification score as weights. 
                    output_prob = torch.softmax(output, dim=1) ## use category prob. as weights or ID prob. as weights?
                    indice = torch.arange(output.size(0))
                    # pdb.set_trace()
                    ########################## category prob level weights.
                    # weights = output_prob[indice, labels]
                    ########################## id/ood level sample weights
                    weights = output_prob[:, :1000].sum(1)
                    # for i in range(image_features.size(0)):
                    #     if labels[i].item() < 1000: 
                    #         ## ID class
                    #         weights[i] = output_prob[i, :1000].sum()
                    #     else:
                    #         weights[i] = output_prob[i, 1000:].sum()

                    for i in range(image_features.size(0)):
                        all_feats[labels[i]].append(image_features[i].unsqueeze(0))
                        all_weights[labels[i]].append(weights[i].unsqueeze(0))

            for i in range(len(all_feats)):
                if len(all_feats[i]) != 0:
                    all_feats[i] = torch.cat(all_feats[i], dim=0)
                    all_weights[i] = torch.cat(all_weights[i], dim=0)
                
                    # pdb.set_trace()
                    # cate_mean = (torch.cat(all_feats[i],dim=0) * torch.cat(all_weights[i], dim=0).unsqueeze(1)).mean(0,keepdim=True)
                    # cate_mean = torch.cat(all_feats[i],dim=0).mean(0,keepdim=True)
                    # cate_mean /= cate_mean.norm(dim=-1, keepdim=True)  ## renorm.  ## how to weight different image features?
                    # all_feats[i] = cate_mean
            all_feats = [x for x in all_feats if isinstance(x, torch.Tensor)]
            all_weights = [x for x in all_weights if isinstance(x, torch.Tensor)]

            all_feats_id = torch.cat(all_feats[:1000], dim=0) ## 11k * 512
            all_weights_id = torch.cat(all_weights[:1000], dim=0) ## 11k * 512
            # pdb.set_trace()
            all_feats_ood = torch.cat(all_feats[1000:], dim=0) ## 11k * 512
            all_weights_ood = torch.cat(all_weights[1000:], dim=0) ## 11k * 512
            self.image_classifier = all_feats_id

            self.image_feat_cache_id = all_feats_id
            self.image_idscore_cache_id = all_weights_id

            self.image_feat_cache_ood = all_feats_ood
            self.image_idscore_cache_ood = all_weights_ood

            self.image_feat_cache = torch.cat((self.image_feat_cache_id, self.image_feat_cache_ood), dim=0)
            self.image_idscore_cache = torch.cat(( self.image_idscore_cache_id,  self.image_idscore_cache_ood), dim=0)
        else:
            pass



    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        # pdb.set_trace()
        image_features, text_features, logit_scale = net(data, return_feat=True)
        # image_classifier = self.image_classifier

        ## image_features: 256*512, 
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention,
        # if len(text_features.shape) == 3: ## 11K*7*512
        #     sim = text_features @ image_features.t()  ## 11k*7*256
        #      #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
        #     sim = torch.exp(-self.beta * (-sim + 1))
        #     temp = sim.unsqueeze(0).transpose(0,-1) * text_features.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
        #     sa_text_features = temp.sum(2) ## 256*11k*512
        #     sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
        #     output = (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        # else:
        #     output = logit_scale * image_features @ text_features.t() # batch * class.
            # output_image = logit_scale * image_features @ image_classifier.t() # batch * class.
        output = logit_scale * image_features @ text_features.t() # batch * class.
        # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(output[:, :class_num], dim=1)

        # _, pred_in_img = torch.max(output_image[:, :class_num], dim=1)
        # ############################### only score in. here is a 
        # output_only_in = output[:, :class_num]
        # output_only_out = output[:, class_num:]
        # score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        # conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
        # cosin_only_in, _ = torch.max(output_only_in, dim=1)
        # ###########
        # print('pay attention, no scale is applied here.')
        pos_logit = output[:, :class_num] ## B*C
        neg_logit = output[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            # pdb.set_trace()
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in = scores.mean(dim=-1)
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # # pdb.set_trace()
        # # self.image_feat_cache N*512
        # # self.image_idscore_cache  N
        # # image_features ## 256*512
        # # cossim = torch.matmul(image_features, self.image_feat_cache.t()) ## 256*N
        # # guided_sim = cossim * self.image_idscore_cache ## 256*N
        # # value, indice = torch.sort(guided_sim, 1) # 256 * N, small 2 large
        # # guided_score = value[:, -self.beta:].mean(1)

        # ############### text NNguided with positive and negtive KNN, with unselected text 
        # ############### 或许可以只根据cos sim 对text id score 进行加权， 总之这里需要再仔细斟酌一下。总的思路是ensemble NN 得到当前test point 的 ood score. 
        # cossim = torch.matmul(image_features, net.text_features_unselected) ## 256*8W
        # value_cos, indice_cos = torch.sort(cossim, 1) # 256 * N, small 2 large

        # cossim_selected = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) ## 256*beta
        # # prob = torch.softmax(cossim_selected / self.tau, dim=1)
        # prob = torch.exp(-self.tau * (1 - cossim_selected)) ## not sentative to the value of beta.
        # prob = prob / prob.sum(1, keepdim=True)
        # guided_score_neartext_modulated = (prob * self.text_idscore_unselected_cache[indice_cos[:, -self.beta:]]).sum(1)


        # guided_score_neardata = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) * self.text_idscore_unselected_cache[indice_cos[:, -self.beta:]] ## 256*N
        # guided_score_neartext = guided_score_neardata.mean(1)

        # guided_sim_id = cossim * self.text_idscore_unselected_cache ## 256*N  #### 这里是否需要加入scaling & softmax??
        # value, indice = torch.sort(guided_sim_id, 1) # 256 * N, small 2 large
        # guided_score_iddata = value[:, -self.beta:].mean(1) ## larger means id

        # guided_sim_ood = cossim * (1 - self.text_idscore_unselected_cache) ## 256*N
        # value, indice = torch.sort(guided_sim_ood, 1) # 256 * N, small 2 large
        # guided_score_ooddata = value[:, -self.beta:].mean(1) ## larger means ood； 这里多个直接求平均值或许可以改成加权和？

        # guided_score_text_mul = guided_score_iddata * (-guided_score_ooddata)  ## score 没有归一化，会有比较大的影响
        # guided_score_text_norm = guided_score_iddata / (guided_score_iddata + guided_score_ooddata)  ### 这个结果差不多，有少量提升;  这里的乘法关系可能也不合适，改成加法。
        # guided_score_text_add = guided_score_iddata - guided_score_ooddata  ### 上面乘法结果更好好一些？ 神奇。。
        

        # # ############### text NNguided with positive and negtive KNN, with selected text 
        # # cossim_id = torch.matmul(image_features, text_features.t()[:, :1000]) ## 256*N
        # # guided_sim_id = cossim_id * self.text_idscore_cache[:1000] ## 256*N
        # # value, indice = torch.sort(guided_sim_id, 1) # 256 * N, small 2 large
        # # guided_score_iddata = value[:, -self.beta:].mean(1) ## larger means id.

        # # cossim_ood = torch.matmul(image_features, text_features.t()[:, 1000:]) ## 256*N
        # # guided_sim_ood = cossim_ood * (1 - self.text_idscore_cache[1000:]) ## 256*N
        # # value, indice = torch.sort(guided_sim_ood, 1) # 256 * N, small 2 large
        # # guided_score_ooddata = value[:, -self.beta:].mean(1) ## larger means ood. 
        # # guided_score_text = guided_score_iddata / (guided_score_iddata + guided_score_ooddata)  ###

        # ############### image NNguided with positive and negtive KNN; here is the focus!!.
        # cossim = torch.matmul(image_features, self.image_feat_cache.t()) ## 256*8W
        # value_cos, indice_cos = torch.sort(cossim, 1) # 256 * N, small 2 large

        # cossim_selected = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) ## 256*beta
        # # prob = torch.softmax(cossim_selected / self.tau, dim=1)
        # prob = torch.exp(-self.tau * (1 - cossim_selected)) ## not sentative to the value of beta.
        # prob = prob / prob.sum(1, keepdim=True)
        # guided_score_nearimg_modulated = (prob * self.image_idscore_cache[indice_cos[:, -self.beta:]]).sum(1)

        # guided_score_neardata = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) * self.image_idscore_cache[indice_cos[:, -self.beta:]] ## 256*N
        # guided_score_nearimg = guided_score_neardata.mean(1)

        # cossim_id = torch.matmul(image_features, self.image_feat_cache_id.t()) ## 256*N
        # guided_sim_id = cossim_id * self.image_idscore_cache_id ## 256*N
        # value, indice = torch.sort(guided_sim_id, 1) # 256 * N, small 2 large
        # guided_score_iddata = value[:, -self.beta:].mean(1) ## larger means id.

        # cossim_ood = torch.matmul(image_features, self.image_feat_cache_ood.t()) ## 256*N
        # guided_sim_ood = cossim_ood * (1 - self.image_idscore_cache_ood) ## 256*N
        # value, indice = torch.sort(guided_sim_ood, 1) # 256 * N, small 2 large
        # guided_score_ooddata = value[:, -self.beta:].mean(1) ## larger means ood. 

        # guided_score_img_norm = guided_score_iddata / (guided_score_iddata + guided_score_ooddata)
        # guided_score_img_mul = guided_score_iddata * (-guided_score_ooddata) 
        # guided_score_img_add = guided_score_iddata - guided_score_ooddata  


        # guided_score = value[:, -self.beta:].mean(1)


        # # print('pay attention, an additional 3.0 scale is applied here.')
        # score_image = torch.softmax(output_image / self.tau, dim=1)
        # conf_in_image = torch.sum(score_image[:, :class_num], dim=1)
        # conf_out_image = torch.sum(score_image[:, class_num:], dim=1)

        # score_merge = score * score_image  
        # conf_in_merge = torch.sum(score_merge[:, :class_num], dim=1)
        # # 

        # max in prob - max out prob
        if self.in_score == 'oodscore' or self.in_score == 'sum':
            conf = conf_in  ## = 1-conf_out
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
