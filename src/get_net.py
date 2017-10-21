import tensorflow as tf
import resnet

def preprocessing(args,images,return_layer,is_training):
    number_of_nets=args.number_of_nets
    features=[i for i in range(number_of_nets)]
    logits=[i for i in range(number_of_nets)]
    with tf.variable_scope(str(number_of_nets)):
        for x in xrange(number_of_nets):
            with tf.variable_scope(str(x)):
                num_classes=None
                if args.loss_function=='l2_loss':    
                    num_classes=1
                elif args.loss_function=='softmax':
                    num_classes=6
                features[x],logits[x] = resnet.inference_small(images,is_training=is_training,num_classes=num_classes)

    if return_layer=='feature':
        return features
    elif return_layer=='final_layer':
        return logits
