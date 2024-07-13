# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




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

    def __init__(self, landsat_inc, sentinel_inc, ln_enc, sn_enc, hidden_dim, sl, sll, sls, out_dim, num_layers, num_heads, ec_dp):
        super(predict_model, self).__init__()

        hidden_size = hidden_dim * sl
        self.fusion = fusion_module(landsat_inc, sentinel_inc, ln_enc, sn_enc, hidden_dim, sl, sll, sls, out_dim, hidden_size, num_layers, num_heads, ec_dp)
                
        self.dropout = nn.Dropout(0.3)         
        self.linear_layer1 = nn.Linear(hidden_dim * sl, (hidden_dim * sl) // 2, bias=True)
        self.linear_layer2 = nn.Linear((hidden_dim * sl) // 2, (hidden_dim * sl) // 4, bias=True)        
        self.output_layer = nn.Linear(hidden_dim // 4, out_dim, bias=True)
        self.layernorm_1 = nn.LayerNorm([sl * hidden_dim], elementwise_affine=False)        
        self.layernorm_2 = nn.LayerNorm((hidden_dim * sl) // 2, elementwise_affine=False)

    def forward(self, X1, X2):
        combined_out = self.fusion(X1, X2)
        out = self.dropout(combined_out)
        out = self.layernorm_1(out)   
        ll1 = self.linear_layer1(out)        
        sigmoid = nn.GELU()
        out = sigmoid(ll1)          
        out = self.layernorm_2(out)              
        ll2 = self.linear_layer2(out)         
        out = sigmoid(ll2)           
        out = out.reshape(out.shape[0], sl, hidden_dim // 4)  
        out = self.output_layer(out)    
        out = sigmoid(out)     
       
        return out

class LSFuseNet(nn.Module):

    def __init__(self, landsat_inc, sentinel_inc, ln_enc, sn_enc, hidden_dim, sl, sll, sls, out_dim, num_layers, num_heads):
        super(LSFuseNet, self).__init__()

        self.fusion_anc = predict_model(landsat_inc, sentinel_inc, ln_enc, sn_enc, hidden_dim, sl, sll, sls, out_dim, num_layers, num_heads, 0.3)
        self.fusion_pos = predict_model(landsat_inc, sentinel_inc, ln_enc, sn_enc, hidden_dim, sl, sll, sls, out_dim, num_layers, num_heads, 0.7)

    def forward(self, X1, X2, X3, X4, X5, X6):
        anc_out = self.fusion_anc(X1, X2)
        pos_out = self.fusion_pos(X3, X4)
        neg_out = self.fusion_anc(X5, X6)
        
        return anc_out, pos_out, neg_out
