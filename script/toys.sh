nohup python run_mminforec.py --lr=0.001 --weight_decay=0 --pred_step=1 --tau=0.1 --data_name=Toys_and_Games --num_hidden_layers=1 --num_attention_heads=1 --attention_probs_dropout_prob=0.5 --hidden_dropout_prob=0.5 --dc_s=1 --dc=1 --num_hidden_layers_gru=1 --mil=4 --loss_fuse_dropout_prob=0.5 --epoch=200 --mem=64 --hidden_size 64 --item_sparsity 50 --ft_epoch 10 --mil_add 6 --cuda 0 --item_sparsity_ratio 0.8 &

mil 为对比视图数

nohup python run_mminforec.py --lr=0.001 --weight_decay=0 --pred_step=1 --tau=0.1 --data_name=Toys_and_Games --num_hidden_layers=1 --num_attention_heads=1 --attention_probs_dropout_prob=0.5 --hidden_dropout_prob=0.5 --dc_s=1 --dc=1 --num_hidden_layers_gru=1 --mil=1 --loss_fuse_dropout_prob=0.5 --epoch=200 --mem=64 --hidden_size 64 --item_sparsity 50 --cuda 6 --ft_epoch 10 --mil_add 0 &
