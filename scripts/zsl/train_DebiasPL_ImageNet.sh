CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8,9 \
python main_DebiasPL_ZeroShot.py \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --cos \
  --epochs 50 \
  --warmup-epoch 5 \
  --trainindex_x CLIPPseudoLabel0.95.csv --trainindex_u CLIPPseudoLabel0.95-Unselected.csv \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.8 \
  --tau 0.4 \
  --CLDLoss \
  --lambda-cld 0.3 \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/ \
  imagenet/