import tensorflow as tf
import numpy as np
import argparse
from model import NTMCopyModel
from utils import generate_random_strings
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--test_seq_length', type=int, default=5)
    parser.add_argument('--model', default="NTM")
    parser.add_argument('--rnn_size', default=128)
    parser.add_argument('--rnn_num_layers', default=3)
    parser.add_argument('--max_seq_length', default=5)
    parser.add_argument('--memory_size', default=128) #this is how many memory locations 
    parser.add_argument('--memory_vector_dim', default=25)
    parser.add_argument('--batch_size', default=5)
    parser.add_argument('--vector_dim', default=8) #binary 8 bit vector  
    parser.add_argument('--shift_range', default=1)
    parser.add_argument('--num_epoches', default=1000000)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--save_dir', default='/home/dl/Documents/shamane-research/One-shot-learning aproach/ntm-master/save')
    parser.add_argument('--tensorboard_dir', default='./summary/copy_task')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    model_list = [NTMCopyModel(args, 1)]
    for seq_length in range(2, args.max_seq_length + 1):  #increasing the sequnce length 
      
        model_list.append(NTMCopyModel(args, seq_length, reuse=True)) #for each sequence this will create the model and the optimizer 

    # model = NTM_model(args, args.max_seq_length)


    with tf.Session() as sess:
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        plt.ion()
        plt.show()
        for b in range(args.num_epoches): #training iteration 
            seq_length = np.random.randint(1, args.max_seq_length + 1)
            model = model_list[seq_length - 1]  #same model which has made to different sequence lengths. 
   
            # seq_length = args.max_seq_length
            x = generate_random_strings(args.batch_size, seq_length, args.vector_dim)
    
            feed_dict = {model.x: x} #creating a feed dictionary, Element in one sequence is 8 bits.
            # print(sess.run([model.state_list, model.output_list], feed_dict=feed_dict))
            if b % 100 == 0:        # test get the validation accuray while trainign 
                p = 0               # select p th sample in the batch to show
                print(x[p, :, :]) #input sequence
                print(sess.run(model.o, feed_dict=feed_dict)[p, :, :]) #output sequence
                state_list = sess.run(model.state_list, feed_dict=feed_dict)
                if args.model == 'NTM':
                    w_plot = []
                    M_plot = np.concatenate([state['M'][p, :, :] for state in state_list]) #concatenate the memory 
                    for state in state_list: #previous state list 
                        w_plot.append(np.concatenate([state['w_list'][0][p, :], state['w_list'][1][p, :]]))
                    plt.imshow(w_plot, interpolation='nearest', cmap='gray')
                    plt.draw()
                    plt.pause(0.001)
                copy_loss = sess.run(model.copy_loss, feed_dict=feed_dict)
                merged_summary = sess.run(model.copy_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                print('batches %d, loss %g' % (b, copy_loss))
            else:                   # train
                sess.run(model.train_op, feed_dict=feed_dict) #gradient decent 
            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)


def test(args):
    print("azzz")
    model = NTMCopyModel(args, args.test_seq_length)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir+'/NTM')
    print(args.save_dir)

    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No Checkpoint found")
        x = generate_random_strings(args.batch_size, args.test_seq_length, args.vector_dim)
        feed_dict = {model.x: x}
        output, copy_loss, state_list = sess.run([model.o, model.copy_loss, model.state_list], feed_dict=feed_dict)
        for p in range(args.batch_size):
            print(x[p, :, :])
            print(output[p, :, :])
        print('copy_loss: %g' % copy_loss)
        if args.model == 'NTM':
            w_plot = []
            for state in state_list:
                w_plot.append(np.concatenate([state['w_list'][0][p, :], state['w_list'][1][p, :]]))
            plt.imshow(w_plot, interpolation='nearest', cmap='gray')
            plt.show()

if __name__ == '__main__':

    main()
