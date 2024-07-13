#!/usr/bin/env python
# coding: utf-8
 
import numpy as np

import pandas as pd
import re
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning.tuner.tuning import Tuner

from einops import rearrange
from torch import einsum

import warnings
import math
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

 
crop = 'corn'
mdl = 'LSFuseNet'
# debug_state = 26

 
x_l = np.load('.../landsat corn.npz') #path to landsat histograms
x_s = np.load('.../sentinel corn.npz') # path to sentinel histograms

 
image_all_l = x_l['output_image']
year_all_l = x_l['output_year']
index_all_l = x_l['output_index']
yield_all_l = x_l['output_yield']

image_all_s = x_s['output_image']
year_all_s = x_s['output_year']
index_all_s = x_s['output_index']
yield_all_s = x_s['output_yield']

 
print(image_all_l.shape)
print(year_all_l.shape)
print(index_all_l[5])
print(yield_all_l.shape)
print('*****')
print(image_all_s.shape)
print(year_all_s.shape)
print(index_all_s[5])
print(yield_all_s.shape)

 
test_years = [2020]
test_yr = test_years[0]
start_yr = 2016
train_yr = list (range(start_yr,test_yr))


val_yr = test_yr-1

print('train_years',train_yr)
print('val_years',val_yr)
print('test_years',test_years)
len(train_yr), len(test_years), test_yr

 
if crop == 'soy':
    timesteps_l=list (range(6,19)) #
    timesteps_s=list (range(7,30))    
else:
    timesteps_l=list (range(5,20)) #
    timesteps_s=list (range(5,33))

timesteps = 21
len(timesteps_l), len(timesteps_s)

 
years_l = year_all_l.astype(int)
years_s = year_all_s.astype(int)

 
lnt_idx = np.where((years_l >= 2016))[0]

 
print(image_all_l.shape)

image_all_l = image_all_l[:,:,timesteps_l,:]
image_all_s = image_all_s[:,:,timesteps_s,:]

image_all_l = image_all_l[lnt_idx,:,:,:]
year_all_l = year_all_l[lnt_idx] 
index_all_l = index_all_l[lnt_idx,:]
yield_all_l = yield_all_l[lnt_idx]

print(image_all_l.shape)
print(year_all_l.shape)
print(index_all_l[5])
print(yield_all_l.shape)


print(image_all_s.shape)
print(year_all_s.shape)
print(index_all_s[5])
print(yield_all_s.shape)

 
def com_val(x,y):
    z = np.intersect1d(x, y)
    
    return z

 
def common_ind(lnt_ind,snt_ind):
    lnt_idx = []
    snt_idx = []

    unq_lnt = np.unique(lnt_ind, axis=0)
    unq_snt = np.unique(snt_ind, axis=0)
    
    lnt_states = unq_lnt[:,0]
    snt_states = unq_snt[:,0]
    
    comm_stt = com_val(lnt_states,snt_states)
    for i in range(0,len(comm_stt)):
        l_idx = np.where(unq_lnt[:,0]== comm_stt[i]) [0]
        s_idx = np.where(unq_snt[:,0]== comm_stt[i]) [0]
        counties_lnt = unq_lnt[l_idx,1]
        counties_snt = unq_snt[s_idx,1]   
        comm_counties = com_val(counties_lnt,counties_snt)
        for j in range(0,len(comm_counties)):
            lnt_id = np.where((lnt_ind[:,0]== comm_stt[i])&(lnt_ind[:,1] == comm_counties[j]))[0]
            snt_id = np.where((snt_ind[:,0]== comm_stt[i])&(snt_ind[:,1] == comm_counties[j]))[0]
            lnt_id = lnt_id.tolist()
            snt_id = snt_id.tolist()            
            lnt_idx = lnt_idx + lnt_id
            snt_idx = snt_idx + snt_id
    
    return lnt_idx,snt_idx

 
lnt_idx,snt_idx  = common_ind(index_all_l,index_all_s)
len(lnt_idx), len(snt_idx)

 
image_all_l = image_all_l[lnt_idx,:,:,:]
year_all_l = year_all_l[lnt_idx] 
index_all_l = index_all_l[lnt_idx,:]
yield_all_l = yield_all_l[lnt_idx]

image_all_s = image_all_s[snt_idx,:,:,:]
year_all_s = year_all_s[snt_idx] 
index_all_s = index_all_s[snt_idx,:]
yield_all_s = yield_all_s[snt_idx]

print(image_all_l.shape)
print(year_all_l.shape)
print(index_all_l[5])
print(yield_all_l.shape)


print(image_all_s.shape)
print(year_all_s.shape)
print(index_all_s[5])
print(yield_all_s.shape)

years_l = year_all_l.astype(int)
years_s = year_all_s.astype(int)

 
train_idx_l = np.where((years_l < test_yr)&(years_l >= start_yr))[0]
train_idx_s = np.where((years_s < test_yr)&(years_s >= start_yr))[0]
print(len(train_idx_l),len(train_idx_s))

val_idx_l = np.where(years_l == val_yr)[0]
val_idx_s = np.where(years_s == val_yr)[0]
print(len(val_idx_l),len(val_idx_s))

# test_idx = np.where(np.isin(years, yrs))[0]
test_idx_l = np.where(years_l == test_yr)[0]
test_idx_s = np.where(years_s == test_yr)[0]
print(len(test_idx_l),len(test_idx_s))

 
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    scores = F.softmax(scores, dim=-1)
    

        
    output = torch.matmul(scores, v)
    return output

 
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)

        
        k = k.to(device)
        q = q.to(device)
        v = v.to(device)        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        mm = torch.matmul(q,k.transpose(2,3))

        mm1 = nn.Softmax(dim=1)

        sf = mm1(mm/math.sqrt(self.d_model))

        mm2 = torch.matmul(sf,v)

        scores = attention(q, k, v, self.d_k)

        concat = scores.transpose(1,2).contiguous()        .view(bs, -1, self.d_model)
        
        output = self.out(concat)

        return output

 
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True)
    return t.softmax(dim = dim)

 
class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.1,
        talking_heads = False,
        prenorm = False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = stable_softmax(sim, dim = -1)
        context_attn = stable_softmax(sim, dim = -2)

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out

 
class sf_cnn(nn.Module):
    def __init__(self, landsat_inc,enc_out_dim,sl,dp_percent):
        super(sf_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = landsat_inc, out_channels = 12, kernel_size =(1,1), stride=1, padding = 0)
        self.pool = nn.MaxPool2d(kernel_size =(2,1), stride =1, padding = (0,0))
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 15, kernel_size =(1,1), stride=1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 15, out_channels = 20, kernel_size =(1,1), stride=1, padding = 0)
        self.dropout = nn.Dropout(dp_percent)

        self.fc1 = nn.Linear(20*29*sl, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, enc_out_dim)
        self.layernorm_1 = nn.LayerNorm(4000, elementwise_affine = False)
        self.layernorm_2 = nn.LayerNorm(2000,  elementwise_affine = False)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)    
        x = F.relu(self.conv2(x))
        x = self.pool(x)   
        x = self.dropout(x)   
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)   
     
        nsamples, nx, ny, nt = x.shape
        out = x.reshape((nsamples,nx*ny*nt))
        ll_1 = self.fc1(out)
        sigmoid = nn.ReLU()
        out = sigmoid(ll_1)
        out = self.layernorm_1(out)        
        ll_2 = self.fc2(out)

        relu = nn.ReLU()
        out = relu(ll_2)
        out = self.layernorm_2(out) 
        out = self.dropout(out)
        x = self.fc3(out)
        dim3 = int(x.shape[1]/nt)
        x = x.reshape(x.shape[0],nt,dim3)
        
        return x

 
class sf_encoder(nn.Module):

	def __init__(self, landsat_inc, enc_out_dim,sl,out_dim):
		super(sf_encoder, self).__init__()


		self.landsat_enc = sf_cnn(landsat_inc, enc_out_dim,sl,0.5)        
		self.lstm_layer = nn.LSTM((enc_out_dim//sl), hidden_dim, n_layers,  batch_first=True, bidirectional = True)
    

	def forward(self, X1):


		landsat_enc_out = self.landsat_enc(X1)   
		out, hidden = self.lstm_layer(landsat_enc_out)
        
		return out

 
class prediction_model(nn.Module):

	def __init__(self, landsat_inc, enc_out_dim,sl,out_dim):
		super(prediction_model, self).__init__()


		self.encoder = sf_encoder(landsat_inc, enc_out_dim,sl,0.5)
		self.decoder = sf_decoder(landsat_inc, enc_out_dim,sl,0.5)


	def forward(self, X1):

		landsat_enc_out = self.encoder(X1)        
		nsamples, nx, ny = landsat_enc_out.shape
		out = landsat_enc_out.reshape((nsamples,nx*ny))        
		out = self.decoder(out)
         
		return out

 
class sf_decoder(nn.Module):
    def __init__(self, landsat_inc,enc_out_dim,sl,dp_percent):
        super(sf_decoder, self).__init__()
        self.nt = sl
        self.conv1 = nn.ConvTranspose2d(in_channels = 20, out_channels = 15, kernel_size =(2,1), stride=1, padding = 0)
        self.pool = nn.MaxPool2d(kernel_size =(2,1), stride =1, padding = (0,0))
        self.conv2 = nn.ConvTranspose2d(in_channels = 15, out_channels = 12, kernel_size =(2,1), stride=1, padding = 0)
        self.conv3 = nn.ConvTranspose2d(in_channels = 12, out_channels = landsat_inc, kernel_size =(2,1), stride=1, padding = 0)
        self.dropout = nn.Dropout(dp_percent)

        self.fc1 = nn.Linear(hidden_dim*sl*2, 2000)
        self.fc2 = nn.Linear(2000, 4000)
        self.fc3 = nn.Linear(4000, 20*61*sl)
        self.layernorm_1 = nn.LayerNorm(2000, elementwise_affine = False)
        self.layernorm_2 = nn.LayerNorm(4000,  elementwise_affine = False)
        
    def forward(self,x):
        
        ll_1 = self.fc1(x)
        sigmoid = nn.ReLU()
        out = sigmoid(ll_1)
        out = self.layernorm_1(out)

        ll_2 = self.fc2(out)
        relu = nn.ReLU()
        out = relu(ll_2)
        out = self.layernorm_2(out)
        x = self.fc3(out)       
        nsamples, nx = x.shape
        out = x.reshape((nsamples,20,61,self.nt))
        
        x = self.dropout(out)     
        x = F.relu(self.conv1(x))
        x = self.dropout(x) 
        x = F.relu(self.conv2(x))        
        x = self.dropout(x)             
        x = F.relu(self.conv3(x))
        
        return x

 
path_to_lnt = '.../path to landsat pretrained encoder'
path_to_snt = '.../path to sentinel pretrained encoder'

pretrained_lnt = torch.load(os.path.join(path_to_lnt, "model_soy_%d.pt"%(235)))
pretrained_snt = torch.load(os.path.join(path_to_snt, "model_soy_%d.pt"%(185)))

 
class fusion_encoder(nn.Module):

    def __init__(self, landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, hidden_size, num_layers,num_heads,ec_dp):
        super(fusion_encoder, self).__init__()

        self.landsat_enc = pretrained_lnt
        self.sentinel_enc = pretrained_snt     
        
        self.lnt_enc_out = ln_enc
        self.snt_enc_out = sn_enc        
        
        self.lndstlyr = nn.Linear(sll*hidden_dim*2,hidden_size,bias = True)
        self.sntlyr   = nn.Linear(sls*hidden_dim*2, hidden_size, bias = True)
        self.attn_ln     = MultiHeadAttention(num_heads, sll*hidden_dim*2)
        self.attn_sn     = MultiHeadAttention(num_heads, sls*hidden_dim*2)
        
    def forward(self, X1, X2):

        
        landsat_enc_out = self.landsat_enc.encoder(X1)     
        sentinel_enc_out = self.sentinel_enc.encoder(X2)       
        nsamples,nt,nxy = landsat_enc_out.shape
        out_ln = landsat_enc_out.reshape((nsamples,nt*nxy))
        nsamples,nt,nxy = sentinel_enc_out.shape
        out_sn = sentinel_enc_out.reshape((nsamples,nt*nxy))        

        outmsa_ln = self.attn_ln(out_ln,out_ln,out_ln,torch.Tensor([1,1])) #MHA  
        outmsa_ln = outmsa_ln.to(device)
        outmsa_sn = self.attn_sn(out_sn,out_sn,out_sn,torch.Tensor([1,1])) #MHA  
        outmsa_sn = outmsa_sn.to(device)  
        outmsa_ln = outmsa_ln.squeeze(1)
        outmsa_sn = outmsa_sn.squeeze(1)        

        out_ln = self.lndstlyr(outmsa_ln)
        sigmoid = nn.GELU()        
        out_sn = self.sntlyr(outmsa_sn)
        out_ln = sigmoid(out_ln)
        out_sn = sigmoid(out_sn)   
        
        return out_ln, out_sn

 
class BertFusion(nn.Module):
    def __init__(self, landsat_dpa_in, sentinel_dpa_in):
        super().__init__()
        self.fusion_function = 'softmax'

    def forward(
        self,
        hidden_states,
        visual_hidden_state,
    ):

        fusion_scores_s = torch.matmul(hidden_states, visual_hidden_state.transpose(-1, -2))
        fusion_scores_l = torch.matmul(visual_hidden_state, hidden_states.transpose(-1, -2))        
        if self.fusion_function == 'softmax':
            fusion_probs_l = nn.Softmax(dim=-1)(fusion_scores_l)
            fusion_probs_s = nn.Softmax(dim=-1)(fusion_scores_s)            
            fusion_output_l = torch.matmul(fusion_probs_l, hidden_states)
            fusion_output_s = torch.matmul(fusion_probs_s, visual_hidden_state)
        
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(dim=-1)
        return fusion_output_l,fusion_output_s

 
class fusion_module(nn.Module):

    def __init__(self, landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, hidden_size, num_layers,num_heads,ec_dp):
        super(fusion_module, self).__init__()

        self.enc = fusion_encoder(landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, hidden_size, num_layers,num_heads,ec_dp)
        
        self.joint_cross_attn = BidirectionalCrossAttention(dim=hidden_size)
        self.linear_layer = nn.Linear(sl*hidden_dim*2,sl*hidden_dim,bias = True)
        self.fusion_noise   = BertFusion(sl*hidden_dim,sl*hidden_dim) 

    def forward(self, X1, X2):
         
        out_lnt, out_snt = self.enc(X1,X2)
        
        
        out_lnt_3d = out_lnt.unsqueeze(0)
        out_snt_3d = out_snt.unsqueeze(0)    
      
        
        attn, crossatn_out_l = self.joint_cross_attn(out_lnt_3d, out_snt_3d) 
        attn, crossatn_out_s = self.joint_cross_attn(out_snt_3d, out_lnt_3d)     
        crossatn_out_l = crossatn_out_l.squeeze(0)
        crossatn_out_s = crossatn_out_s.squeeze(0)
        fusn_nois_lnt, fusn_nois_snt = self.fusion_noise(crossatn_out_l,crossatn_out_s)
        fusion_noise_out  = torch.cat((fusn_nois_lnt,fusn_nois_snt), dim = 1)
        
        ll = self.linear_layer(fusion_noise_out)        
        sigmoid = nn.GELU()
        combined_out = sigmoid(ll)        
              
        return combined_out

 
class predict_model(nn.Module):

    def __init__(self, landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, num_layers,num_heads,ec_dp):
        super(predict_model, self).__init__()

        hidden_size = hidden_dim*sl
        self.fusion = fusion_module(landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, hidden_size, num_layers,num_heads,ec_dp)
                
        self.dropout = nn.Dropout(0.3)         
        self.linear_layer1 = nn.Linear(hidden_dim*sl,(hidden_dim*sl)//2,bias = True)
        self.linear_layer2 = nn.Linear((hidden_dim*sl)//2,(hidden_dim*sl)//4,bias = True)        
        self.output_layer = nn.Linear(hidden_dim//4, out_dim, bias = True)
        self.layernorm_1 = nn.LayerNorm([sl*hidden_dim], elementwise_affine = False)        
        self.layernorm_2 = nn.LayerNorm((hidden_dim*sl)//2, elementwise_affine = False)
        

    def forward(self, X1, X2):

        
        combined_out = self.fusion(X1,X2)
        out = self.dropout(combined_out)
        out = self.layernorm_1(out)   
        ll1 = self.linear_layer1(out)        
        sigmoid = nn.GELU()
        out = sigmoid(ll1)          
        out = self.layernorm_2(out)              
        ll2 = self.linear_layer2(out)         
        out = sigmoid(ll2)           
        out = out.reshape(out.shape[0],sl,hidden_dim//4)  
        out = self.output_layer(out)    
        out = sigmoid(out)     
       
        return out

 
class contrastive_model(nn.Module):

    def __init__(self, landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, num_layers,num_heads):
        super(contrastive_model, self).__init__()

        self.fusion_anc = predict_model(landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, num_layers,num_heads,0.3)
        self.fusion_pos = predict_model(landsat_inc, sentinel_inc, ln_enc,sn_enc,hidden_dim, sl,sll,sls,out_dim, num_layers,num_heads,0.7)
    def forward(self, X1, X2,X3,X4,X5,X6):

        
        anc_out = self.fusion_anc(X1,X2)
        pos_out = self.fusion_pos(X3,X4)
        neg_out = self.fusion_anc(X5,X6)
        
        return anc_out,pos_out,neg_out

 
train_images_l = image_all_l[train_idx_l]
val_images_l = image_all_l[val_idx_l]
test_images_l = image_all_l[test_idx_l]

train_images_s = image_all_s[train_idx_s]
val_images_s = image_all_s[val_idx_s]
test_images_s = image_all_s[test_idx_s]

train_yld = yield_all_l[train_idx_l]
val_yld = yield_all_l[val_idx_l]
test_yld = yield_all_l[test_idx_l]

train_loc_l = index_all_l[train_idx_l]
val_loc_l = index_all_l[val_idx_l]
test_loc_l = index_all_l[test_idx_l]

train_loc_s = index_all_s[train_idx_s]
val_loc_s = index_all_s[val_idx_s]
test_loc_s = index_all_s[test_idx_s]

train_yr_l = year_all_l[train_idx_l]
val_yr_l = year_all_l[val_idx_l]
test_yr_l = year_all_l[test_idx_l]

train_yr_s = year_all_s[train_idx_s]
val_yr_s = year_all_s[val_idx_s]
test_yr_s = year_all_s[test_idx_s]



print('train_images_l.shape',train_images_l.shape)
print('test_images_l.shape',test_images_l.shape)

print('train_images_s.shape',train_images_s.shape)
print('test_images_s.shape',test_images_s.shape)


print('train_yld.shape',train_yld.shape)
print('test_yld.shape',test_yld.shape)

 
class CropDataset_train(Dataset):

    def __init__(self, x, y, p, q, s, t):


        x = np.transpose(x, (0,3,1,2))
        y = np.transpose(y, (0,3,1,2))

        x_1 = torch.Tensor(x)
        x_2 = torch.flip(x_1, dims=(3,))
        x_3 = torch.cat((x_1, x_2), 0)
        x_4 = torch.cat((x_1, x_2), 0)        
        x_ = torch.cat((x_3, x_4), 0)
        
        y_1 = torch.Tensor(y)
        y_2 = torch.Tensor(y) 
        y_3 = torch.cat((y_1, y_2), 0)
        y_4 = torch.flip(y_3, dims=(3,))       
        y_ = torch.cat((y_3, y_4), 0)

        
        p_1 = torch.Tensor(p)
        p_2 = torch.Tensor(p)
        p_3 = torch.cat((p_1, p_2), 0)
        p_4 = torch.cat((p_1, p_2), 0)
        p_ = torch.cat((p_3, p_4), 0)
        
        
        q_1 = torch.Tensor(q)
        q_2 = torch.Tensor(q)
        q_3 = torch.cat((q_1, q_2), 0)
        q_4 = torch.cat((q_1, q_2), 0)
        q_ = torch.cat((q_3, q_4), 0)
        
        
        s_1 = torch.Tensor(s)
        s_2 = torch.Tensor(s)
        s_3 = torch.cat((s_1, s_2), 0)
        s_4 = torch.cat((s_1, s_2), 0)
        s_ = torch.cat((s_3, s_4), 0)       
        
        
        t_1 = torch.Tensor(t)
        t_2 = torch.Tensor(t)
        t_3 = torch.cat((t_1, t_2), 0)
        t_4 = torch.cat((t_1, t_2), 0)
        t_ = torch.cat((t_3, t_4), 0)        

        print(x_.shape, y_.shape)
        self.X = x_
        self.Y = y_
        self.P = p_
        self.Q = q_
        self.S = s_
        self.T = t_

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx],  self.P[idx],self.Q[idx],self.S[idx],  self.T[idx]

 
class CropDataset(Dataset):

    def __init__(self, x, y, p, q, s, t):


        x = np.transpose(x, (0,3,1,2))
        y = np.transpose(y, (0,3,1,2))

        x_ = torch.Tensor(x)
        y_ = torch.tensor(y)
        p_ = torch.Tensor(p)
        q_ = torch.Tensor(q)
        s_ = torch.Tensor(s)
        t_ = torch.Tensor(t)
        print(x_.shape, y_.shape, t_.shape)
        self.X = x_
        self.Y = y_
        self.P = p_
        self.Q = q_
        self.S = s_
        self.T = t_

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx],  self.P[idx],self.Q[idx],self.S[idx],  self.T[idx]

 
train_data = CropDataset(train_images_l,train_images_s, train_loc_l, train_loc_s, train_yr_l, train_yld)
val_data = CropDataset(val_images_l,val_images_s, val_loc_l, val_loc_s, val_yr_l,val_yld)
test_data = CropDataset(test_images_l,test_images_s, test_loc_l, test_loc_s, test_yr_l, test_yld)

 
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

 
# GPU related info
cuda = 1
device = torch.device("cuda:1" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

 
path_to_save = '.../path to save models'
path_to_save

 
def evaluate(model,data_loader, dataset):
    model.eval()
    
    eval_loss1 = float(0)

    correct = 0
    avg_loss = []
    output_list = []
    target_list = []
    count = 0
    with torch.no_grad():
        for data_l, data_s, loc_l,loc_s,yr_l, target in data_loader:
            data_l, data_s, loc_l,loc_s,yr_l, target = (data_l.float()).to(device),(data_s.float()).to(device),(loc_l.float()).to(device),(loc_s.float()).to(device),(yr_l.float()).to(device),target.float().to(device)
            output, output_pos, output_neg = model(data_l,data_s,data_l,data_s,data_l,data_s)

            target = target.repeat(output.shape[1], 1).T
            output = output.squeeze(-1)        
            eval_lossb1 = criterion_pred(output, target).mean(axis=0)*len(target)
            eval_lossb1 = torch.tensor(eval_lossb1)    
            eval_loss1 += eval_lossb1      
            count += len(target)
            output_list += output.tolist()
            target_list += target.tolist()

    eval_loss1 /= count
    eval_loss1 = torch.sqrt(torch.tensor(eval_loss1))    
    eval_loss1 = eval_loss1.cpu().numpy()
    if dataset == 'Validation':
        print('{} set: loss1: {} \n'.format(dataset, eval_loss1))  
        return eval_loss1, output_list, target_list

    if dataset == 'Test':
        print('{} set: loss1: {} \n'.format(dataset, eval_loss1))  
        return eval_loss1, output_list, target_list

 
def make_pairs(l_loc,s_loc,l_yr,indix_l,indix_s,year_l):
    pos_idxs = []
    neg_idxs = []
    total_idx = list(range(0,len(indix_l)))
    
    for i in range(0,len(indix_l)):
        posidxs = torch.where((indix_l[:,0] == l_loc[0]) &(indix_l[:,1] == l_loc[1]) &(year_l[:] == l_yr))[0]
        posidxs = posidxs.cpu().numpy()            
        pos_id  =  np.random.choice(posidxs)
        negIdxs = torch.where((indix_l[:,0] != l_loc[0]) &(indix_l[:,1] != l_loc[1]))[0]
        negIdxs = negIdxs.cpu().numpy()
        neg_id  =  np.random.choice(negIdxs)
       
    return [pos_id, neg_id]

 
def train(model, optimizer, train_loader, eval_loader,epcs):
	cl_loss1= []
	cl_loss2= [] 
	cls_loss= []     
	mse_loss = []
    
	history_train = []
	history_val1 = []
	history_val2 = []
   
	output_train = []
	target_train = []

	for epoch in range(1, epcs+1):
		model.train()
      
		for batch_idx, (data_l, data_s, loc_l,loc_s,yr_l, target) in enumerate(train_loader):

			data_l, data_s, loc_l,loc_s,yr_l, target = (data_l.float()).to(device),(data_s.float()).to(device),(loc_l.float()).to(device),(loc_s.float()).to(device),(yr_l.float()).to(device),target.float().to(device)
			optimizer.zero_grad(set_to_none = True)                        
			positive_pairs = []
			negative_pairs = []                        
			for i in range(0,len(target)):
			    positive_idx, negative_idx = make_pairs(loc_l[i],loc_s[i], yr_l[i],loc_l,loc_s,yr_l)            
			    positive_pairs.append(positive_idx)
			    negative_pairs.append(negative_idx)                
			data_l_pos = data_l[positive_pairs]
			data_s_pos = data_s[positive_pairs]
			lnt_pos = data_l_pos
			snt_pos = data_s_pos     
			anc_output, positive_out, negative_out= model(data_l, data_s,lnt_pos, snt_pos,data_l[negative_pairs], data_s[negative_pairs])  
			target = target.repeat(anc_output.shape[1], 1).T
			output = anc_output.squeeze(-1) 
            
			loss_1 = torch.sqrt(criterion_pred(output,target))
			loss_2 = criterion_triplet(anc_output, positive_out,target)
			loss_3 = criterion_triplet(anc_output, negative_out,target)  
        
			loss = loss_1 +loss_2 +loss_3
 
			loss.backward()
			optimizer.step()
		if epoch % 2 == 0:
		  torch.save(model, os.path.join(path_to_save, "model_soy_%d.pt"%(epoch)))   

		print('Train Epoch: {} Training Loss: {}'.format(epoch,loss.item()))
		result_val1,train_preds, labels = evaluate(model,eval_loader,'Validation')
		mse_loss.append(loss_1.item()) 
    
		history_train.append(loss.item())
		history_val1.append(result_val1)        
		output_train.append(train_preds)
		target_train.append(labels) 
        
	return history_train, history_val1,output_train, target_train, mse_loss 
    

 
landsat_inc = 9
sentinel_inc = 13 
hidden_dim = 40

sl = timesteps
sls = len(timesteps_s)
sll = len(timesteps_l)


ln_enc  = 64*landsat_inc*sll 
sn_enc = 64*sentinel_inc*sls 

num_heads = 4
out_dim = 1
epochs = 251
num_layers = 1

 
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform(module.weight)       
    if type(module) == nn.Conv2d:    
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.01)

 
class MarginBasedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginBasedContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(torch.clamp(target * euclidean_distance, min=0.0, max=self.margin))
        return loss

 
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
        mask = (diff < self.delta).float()
        loss = mask * mse_loss + (1 - mask) * self.delta * diff - 0.5 * (1 - mask) * self.delta ** 2
        return torch.mean(loss)

 
model_pred = contrastive_model(landsat_inc,sentinel_inc,ln_enc,sn_enc,hidden_dim,sl,sll,sls,out_dim,num_layers,num_heads).to(device)
model_pred = model_pred.apply(init_weights)


criterion_pred = nn.MSELoss().to(device)
criterion_hub = HuberLoss(delta=1.0).to(device)
criterion_print = nn.MSELoss(reduce = False, reduction = 'None').to(device)
criterion_triplet = MarginBasedContrastiveLoss(margin=0.5)

params = list(model_pred.parameters()) 
optimizer = optim.SGD(params, lr=0.000071, weight_decay = 0.001, momentum = 0.70)

 
history_t, history_v1, output_train, target_train, ms_l  = train(model_pred,optimizer, train_dataloader, val_dataloader,epochs)

 
plt.figure(figsize=(8,6))
plt.title('Training Loss vs Epochs')
plt.plot([x for x in history_t])

plt.figure(figsize=(8,6))
plt.title('Validation Loss vs Epochs')
plt.plot([x for x in history_v1])

 
losses1 = []

mdls = []
for m in range(2,epochs+1,2):
    model_trained=torch.load(os.path.join(path_to_save, "model_soy_%d.pt"%(m)))
    loss1, preds, labels = evaluate(model_trained, test_dataloader, 'Test')
    losses1.append(loss1)

    mdls.append(m)

print(mdls[losses1.index(min(losses1))], min(losses1) )

 
best = mdls[losses1.index(min(losses1))]
best

 
model_trained=torch.load(os.path.join(path_to_save, "model_soy_%d.pt"%(best)))
loss1,  preds, labels = evaluate(model_trained, test_dataloader, 'Test')
loss_met_0 = loss1

print('loss_met_0', loss_met_0)

 
fin_label = np.array(labels)
fin_label = fin_label[:,-1]
fin_preds = np.array(preds)
fin_preds = fin_preds[:,-1]

 
df_test = pd.DataFrame(test_loc_l, columns=['State','County'])
df_test['Yield'] = test_yld
df_test['Atual'] = fin_label
df_test['Predicted'] = fin_preds
df_test[0:12]

 
# import sklearn
from sklearn import metrics
from sklearn.metrics import r2_score 

rsquare_vis = r2_score(fin_label,fin_preds)
mae_vis = metrics.mean_absolute_error(fin_label,fin_preds)

from sklearn.metrics import mean_squared_error
rmse = (metrics.mean_squared_error(fin_label,fin_preds))/529
rmse

 
loss_1 = loss1
loss_1

 
rmse_df = pd.DataFrame({'loss_1': loss_1, 'r_square':rsquare_vis,'mae':mae_vis}, index = [crop+'_'+mdl+'_'+str(test_yr)])
rmse_df

 
rmse_df_ep = pd.DataFrame({'loss_1': loss_met_0}, index = ['rmse2_0_'+crop+'_'+mdl+'_'+str(test_yr)+'-0'])


rmse_df_ep

 
df_test.to_csv('.../Predictions.csv') # path to save prediction file
rmse_df.to_csv('.../error.csv') # path to save error file
rmse_df_ep.to_csv('.../early prediction error.csv') # path to save early prediction error file

print('file created '+crop+'_'+mdl+'_'+str(test_yr))

 


 


 


