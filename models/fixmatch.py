import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, base_encoder, num_classes, norm_layer=None):
        super(ResNet, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer)
        assert not hasattr(self.backbone, 'fc'), "fc should not in backbone"
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class FixMatch(nn.Module):

    def __init__(self, base_encoder, num_classes=1000, eman=False, momentum=0.999, norm=None, num_crops=0):
        super(FixMatch, self).__init__()
        self.eman = eman
        self.momentum = momentum
        self.num_crops = num_crops
        self.main = ResNet(base_encoder, num_classes, norm_layer=norm)
        # build ema model
        if eman:
            print("using EMAN as techer model")
            self.ema = ResNet(base_encoder, num_classes, norm_layer=norm)
            for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_main.data)  # initialize
                param_ema.requires_grad = False  # not update by gradient
        else:
            self.ema = None

    def momentum_update_ema(self):
        state_dict_main = self.main.state_dict()
        state_dict_ema = self.ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)

    def forward(self, im_x, im_u_w=None, im_u_s=None, im_u_s2=None, mc_imgs_w=None, mc_imgs_s=None):
        if im_u_w is None and im_u_s is None:
            logits = self.main(im_x)
            return logits

        batch_size_x = im_x.shape[0]
        if self.num_crops > 0:
            if not self.eman:            
                inputs = torch.cat((im_x, im_u_w, im_u_s))
                logits = self.main(inputs)
                logits_x = logits[:batch_size_x]
                logits_u_w, logits_u_s = logits[batch_size_x:].chunk(2)

                logit_mc_list = [self.main(mc_img) for mc_img in mc_imgs]
                return logits_x, logits_u_w, logits_u_s, logit_mc_list

            else:
                # use ema model for pesudo labels
                inputs = torch.cat((im_x, im_u_s))
                logits = self.main(inputs)
                logits_x = logits[:batch_size_x]
                logits_u_s = logits[batch_size_x:]
                
                logit_mc_list_w, logit_mc_list_s = None, None
                with torch.no_grad():  # no gradient to ema model
                    logits_u_w = self.ema(im_u_w)
                    if mc_imgs_w is not None:
                        logit_mc_list_w = [self.ema(mc_img) for mc_img in mc_imgs_w]

                if mc_imgs_s is not None:
                    logit_mc_list_s = [self.main(mc_img) for mc_img in mc_imgs_s]
                return logits_x, logits_u_w, logits_u_s, logit_mc_list_w, logit_mc_list_s
        else:
            if im_u_s2 is not None:
                if not self.eman:
                    inputs = torch.cat((im_x, im_u_w, im_u_s, im_u_s2))
                    logits = self.main(inputs)
                    logits_x = logits[:batch_size_x]
                    logits_u_w, logits_u_s, logits_u_s2 = logits[batch_size_x:].chunk(3)
                else:
                    # use ema model for pesudo labels
                    inputs = torch.cat((im_x, im_u_s, im_u_s2))
                    logits = self.main(inputs)
                    logits_x = logits[:batch_size_x]
                    logits_u_s, logits_u_s2 = logits[batch_size_x:].chunk(2)
                    with torch.no_grad():  # no gradient to ema model
                        logits_u_w = self.ema(im_u_w)

                return logits_x, logits_u_w, logits_u_s, logits_u_s2
            else:
                if not self.eman:
                    inputs = torch.cat((im_x, im_u_w, im_u_s))
                    logits = self.main(inputs)
                    logits_x = logits[:batch_size_x]
                    logits_u_w, logits_u_s = logits[batch_size_x:].chunk(2)
                else:
                    # use ema model for pesudo labels
                    inputs = torch.cat((im_x, im_u_s))
                    logits = self.main(inputs)
                    logits_x = logits[:batch_size_x]
                    logits_u_s = logits[batch_size_x:]
                    with torch.no_grad():  # no gradient to ema model
                        logits_u_w = self.ema(im_u_w)

                return logits_x, logits_u_w, logits_u_s


def get_fixmatch_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        FixMatch model
    """
    if isinstance(model, str):
        model = {
            "FixMatch": FixMatch,
        }[model]
    return model
