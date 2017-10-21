for i in 5
do
echo $i
python extract.py --net_number $i --test_set val --loss softmax --model_path "weights/"$i"nets_model.npy" --data_path ../HappeiDetectedFaces/validation_all_faces/
done