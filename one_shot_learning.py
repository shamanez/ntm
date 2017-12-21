from utils import OmniglotDataLoader, one_hot_decode, five_hot_decode
import tensorflow as tf
import argparse
import numpy as np
from model import NTMOneShotLearningModel
from tensorflow.python import debug as tf_debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--label_type', default="one_hot", help='one_hot or five_hot')
    parser.add_argument('--n_classes', default=5) #number of classes per one epicsode
    parser.add_argument('--seq_length', default=50)  #how many images per one episode 
    parser.add_argument('--augment', default=True) # ?
    parser.add_argument('--model', default="MANN", help='LSTM, MANN, MANN2 or NTM')
    parser.add_argument('--read_head_num', default=4) #reading from the memory with 4 heards.
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_epoches', default=100000)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--rnn_size', default=200)   #controller size 
    parser.add_argument('--image_width', default=20) #we squash the image 
    parser.add_argument('--image_height', default=20)
    parser.add_argument('--rnn_num_layers', default=1) #number of layers in the RNN
    parser.add_argument('--memory_size', default=128)#How many elements in the external memmory
    parser.add_argument('--memory_vector_dim', default=40)
    parser.add_argument('--shift_range', default=1, help='Only for model=NTM') #inorder to circular convolution location based 
    parser.add_argument('--write_head_num', default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)') #this is for the write head 
    parser.add_argument('--test_batch_num', default=100)
    parser.add_argument('--n_train_classes', default=700) #we devide classes 
    parser.add_argument('--n_test_classes', default=264) #we test 423 outof those classes 
    parser.add_argument('--save_dir', default='./save/one_shot_learning')
    parser.add_argument('--tensorboard_dir', default='./summary/one_shot_learning')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    model = NTMOneShotLearningModel(args)
    
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )

    with tf.Session() as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()

        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
        #print(args)
        #print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")
        for b in range(10000000):

            print("Inside the Iteration")

            # Test

            if b % 100 == 0:          #this is more of validating the model
                x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,  #we do not optimize this just take the loss
                                                              type='test',
                                                              augment=args.augment,
                                                              label_type=args.label_type)
                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}  #feeding to the algorithm
                output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict) #get the output predicted  also training list

                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)  #this is to tensorbards 
                train_writer.add_summary(merged_summary, b)

                # state_list = sess.run(model.state_list, feed_dict=feed_dict)  # For debugging
                # with open('state_long.txt', 'w') as f:
                #     print(state_list, file=f)
                accuracy = test_f(args, y, output)   #getting the accuracy need to inpit the output of the network and read output y
                for accu in accuracy:
                    print(accu)
                print((b, learning_loss))

            # Save model

            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

            # Train

            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='train',
                                                          augment=args.augment,
                                                          label_type=args.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            learning_loss = sess.run([model.learning_loss], feed_dict=feed_dict)
            sess.run(model.train_op, feed_dict=feed_dict)
            print(learning_loss,"Learing losss")


def test(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
        y_list = []
        output_list = []
        loss_list = []
        for b in range(args.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='test',
                                                          augment=args.augment,
                                                          label_type=args.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
            y_list.append(y)
            output_list.append(output)
            loss_list.append(learning_loss)
        accuracy = test_f(args, np.concatenate(y_list, axis=0), np.concatenate(output_list, axis=0))
        for accu in accuracy:
            print(accu)
        print(np.mean(loss_list))


def test_f(args, y, output):

    correct = [0] * args.seq_length   #correctly predicted 
    
    total = [0] * args.seq_length #total predicted
    if args.label_type == 'one_hot':
        y_decode = one_hot_decode(y)  #getting the index of the arg max 
        output_decode = one_hot_decode(output) #getting the index of argmax

    elif args.label_type == 'five_hot':
        y_decode = five_hot_decode(y)
        output_decode = five_hot_decode(output)

    for i in range(np.shape(y)[0]): #this is iterating through the each predicted example in the batch

        y_i = y_decode[i]  #one y_i have 50 elementns 
        print("Printing the correct classes of the sequence",y_i)
        output_i = output_decode[i]
             # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):  #now for each time step we iterate. 
            print(j)
            if y_i[j] not in class_count:   #get the first class in the starting time step. Check whether the sequence saw it before
                class_count[y_i[j]] = 0  #start for the that class with sero
            class_count[y_i[j]] += 1  #add one for the class    #each time when this sees a class it will up the counts
            print("Printing the class counts",class_count)
            print("printing the class cout of the current correct-class in the sequence",class_count[y_i[j]])
            total[class_count[y_i[j]]] += 1
            print(total)

            if y_i[j] == output_i[j]:  #if corerctly predicted the current time step one
                correct[class_count[y_i[j]]] += 1 #This is to basically find how many times networks see a class and how many times network correctly predicted a class. 
            print("Printing the correctness thing",correct)
       
    #basically here we calculate end of each time step how many times I have seen this examples and how many times my network predicted correctly.

    #here total is a [0,8,2,3,......49]  of there  8 in second position is in the batch the network has seen same class for twice while and . 
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)] # accuracy is get by how many time steps in a back has seen sa


if __name__ == '__main__':
    main()