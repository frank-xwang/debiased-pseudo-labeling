import torch
import torch.nn as nn
from utils import dist_utils


def build_hidden_head(num_mlp, dim_mlp, dim):
    modules = []
    for _ in range(1, num_mlp):
        modules.append(nn.Linear(dim_mlp, dim_mlp))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(dim_mlp, dim))
    return nn.Sequential(*modules)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, num_mlp=2, norm_layer=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        assert num_mlp >= 1
        if norm_layer is nn.BatchNorm2d or norm_layer is None:
            self.do_shuffle_bn = True
        else:
            self.do_shuffle_bn = False

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(norm_layer=norm_layer)
        self.encoder_k = base_encoder(norm_layer=norm_layer)

        dim_mlp = self.encoder_q.out_channels

        self.encoder_q.fc = build_hidden_head(num_mlp, dim_mlp, dim)
        self.encoder_k.fc = build_hidden_head(num_mlp, dim_mlp, dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = dist_utils.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # for inference, query model only
        if im_k is None:
            feats = self.encoder_q(im_q)
            return feats

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.encoder_q.fc(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.do_shuffle_bn:
                im_k, idx_unshuffle = dist_utils.batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.do_shuffle_bn:
                k = dist_utils.batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCoEMAN(MoCo):

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, num_mlp=2, norm_layer=None):
        super(MoCoEMAN, self).__init__(base_encoder, dim, K, m, T, num_mlp, norm_layer)
        self.do_shuffle_bn = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder's state_dict. In MoCo, it is parameters
        """
        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for (k_q, v_q), (k_k, v_k) in zip(state_dict_q.items(), state_dict_k.items()):
            assert k_k == k_q, "state_dict names are different!"
            if 'num_batches_tracked' in k_k:
                v_k.copy_(v_q)
            else:
                v_k.copy_(v_k * self.m + (1. - self.m) * v_q)


def get_moco_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        MoCo model
    """
    if isinstance(model, str):
        model = {
            "MoCo": MoCo,
            "MoCoEMAN": MoCoEMAN,
        }[model]
    return model
