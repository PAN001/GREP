i=5
python LSTM_train.py --net_number  $i --train_path "features/train_"$i"nets_feature3.npy" --val_path "features/val_"$i"nets_feature3.npy" --iter 4

# python LSTM_train.py --net_number $i --train_path "features/train_"$i"nets_feature3.npy" --val_path "features/val_"$i"nets_feature3.npy" --weights weights/5nets_lstm_model.npy --iter 4