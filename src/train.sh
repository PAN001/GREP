for i in 5
do
echo $i
# python extract.py --net_number $i --test_set train --loss softmax --model_path "weights/"$i"nets_model.npy" --data_path ../HappeiDetectedFaces/train_all_faces/
python LSTM_train.py --net_number  $i --train_path "features/train_"$i"nets_feature_correct.npy" --val_path "features/val_"$i"nets_feature_correct.npy" --iter 4 --weights weights/5nets_lstm_model.npy
done
