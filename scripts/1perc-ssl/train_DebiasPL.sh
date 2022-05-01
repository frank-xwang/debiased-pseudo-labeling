CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_DebiasPL.py \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --cos \
  --epochs 50 \
  --warmup-epoch 5 \
  --trainindex_x train_1p_index.csv --trainindex_u train_99p_index.csv \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.8 \
  --tau 0.4 \
  --CLDLoss \
  --lambda-cld 0.3 \
  --multiviews \
  --qhat_m 0.999 \
  --output checkpoints/ \
  imagenet/