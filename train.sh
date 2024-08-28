CUDA_VISIBLE_DEVICES="0,1" \
python train.py \
--hiera_path "/kaggle/input/segment/sam2_hiera_large.pt" \
--train_image_path "" \
--train_mask_path "<set your training mask dir here>" \
--save_path "<set your checkpoint saving dir here>" \
--epoch 20 \
--lr 0.001 \
--batch_size 12
