import tensorflow as tf
import numpy as np

class MANNCell():
    def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma=0.95,
                 reuse=False, k_strategy='separate'):
        self.rnn_size = rnn_size #size of the controller
        self.memory_size = memory_size #size of the memory
        self.memory_vector_dim = memory_vector_dim
        self.head_num = head_num                                    # #(read head) == #(write head)
        self.reuse = reuse
        self.controller = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size) #controller as a BasicLSTMCell
        self.step = 0
        self.gamma = gamma
        self.k_strategy = k_strategy

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state['read_vector_list']      # read vector (the content that is read out, length = memory_vector_dim)
        prev_controller_state = prev_state['controller_state']      # state of controller (LSTM hidden state)

        # x + prev_read_vector -> controller (RNN) -> controller_output

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1) #concatonate with the precior read vec
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_controller_state)  #put them in the basic LSTM

        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M)
        #                       -> a (dim = memory_vector_dim, add vector, only when k_strategy='separate')
        #                       -> alpha (scalar, combination of w_r and w_lu)

        if self.k_strategy == 'summary':  #two types ok controller outputs.
            num_parameters_per_head = self.memory_vector_dim + 1
        elif self.k_strategy == 'separate':
            num_parameters_per_head = self.memory_vector_dim * 2 + 1
        total_parameter_num = num_parameters_per_head * self.head_num  #total parameters   #4 read heads 
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            o2p_w = tf.get_variable('o2p_w', [controller_output.get_shape()[1], total_parameter_num],
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            o2p_b = tf.get_variable('o2p_b', [total_parameter_num],
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            parameters = tf.nn.xw_plus_b(controller_output, o2p_w, o2p_b)  #this gives the number of paramers 
        head_parameter_list = tf.split(parameters, self.head_num, axis=1) #split parameters by number of head 

        # k, prev_M -> w_r
        # alpha, prev_w_r, prev_w_lu -> w_w

        prev_w_r_list = prev_state['w_r_list']      # vector of weightings (blurred address) over locations
        prev_M = prev_state['M']  #this is the previous memory matric
        prev_w_u = prev_state['w_u']  #previous used weights , these things should have a value. This depends on 

#here we need previous least used weight matrix and indices 
        prev_indices, prev_w_lu = self.least_used(prev_w_u)  #this gives the orevious used weughts , get the previous used weigts and calculate the leasted used one eq-21
        w_r_list = []  #read vector list 
        w_w_list = [] #write vector lost 
        k_list = [] 
        a_list = []
        # p_list = []   # For debugging
        for i, head_parameter in enumerate(head_parameter_list):  #iterate 5 times since we have 5 heads 
         
            with tf.variable_scope('addressing_head_%d' % i):
                k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim], name='k') #get first set of elements up to vector dim to take k paramters 
                if self.k_strategy == 'separate':
                    a = tf.tanh(head_parameter[:, self.memory_vector_dim:self.memory_vector_dim * 2], name='a') #this is like another paramaters set 
           
                sig_alpha = tf.sigmoid(head_parameter[:, -1:], name='sig_alpha') #use this paramter to compute the comptex combination of previously read weights and least used weights
                w_r = self.read_head_addressing(k, prev_M) #this is for calculating the cosin similirity and obtain the 

            #here to get the write head adressing we need to input scalar for sigmoid  , previously read weights(That is least used locations) , and other one is previously least used weights 
                w_w = self.write_head_addressing(sig_alpha, prev_w_r_list[i], prev_w_lu) #this is the write head which calculate using previous right list and previous least used
            w_r_list.append(w_r)  #append read weight list by cosine similarity and then normalize with softmaz 
            w_w_list.append(w_w) #apend the write weight lists. 
            k_list.append(k)  #this is for the output to preduct the class 
            if self.k_strategy == 'separate':
                a_list.append(a)  #This is like in the NTM case a sepretate param list to predict the outptut
            # p_list.append({'k': k, 'sig_alpha': sig_alpha, 'a': a})   # For debugging

        w_u = self.gamma * prev_w_u + tf.add_n(w_r_list) + tf.add_n(w_w_list)   # eq (20) #for the used waights we need previous used wughts , previouslt read weifhts new right weguths   

        # Set least used memory location computed from w_(t-1)^u to zero
      
        M_ = prev_M * tf.expand_dims(1. - tf.one_hot(prev_indices[:, -1], self.memory_size), dim=2)#this is more like erasing the least used location

        # Writing

        M = M_
        with tf.variable_scope('writing'):
            for i in range(self.head_num):
                w = tf.expand_dims(w_w_list[i], axis=2)
                if self.k_strategy == 'summary':    
                    k = tf.expand_dims(k_list[i], axis=1) #we use the same k to write to the memory  #this is same as in the paper. 
                elif self.k_strategy == 'separate':
                    k = tf.expand_dims(a_list[i], axis=1) #here we use differet vector like a write vector like in the ntm to use
                M = M + tf.matmul(w, k)

        # Reading

        read_vector_list = []
        with tf.variable_scope('reading'):
            for i in range(self.head_num):
                read_vector = tf.reduce_sum(tf.expand_dims(w_r_list[i], dim=2) * M, axis=1)  #for each head getting the read vecrtor list.
                read_vector_list.append(read_vector)

        # controller_output -> NTM output

        NTM_output = tf.concat([controller_output] + read_vector_list, axis=1)     #here for the NTM CELL output we also supply the currect time read vector leess 
#here the controller output is the same output.
        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_r_list': w_r_list,
            'w_w_list': w_w_list,
            'w_u': w_u,
            'M': M,
        }

        self.step += 1
        return NTM_output, state 

    def read_head_addressing(self, k, prev_M): #K paramaters and previous memory matraic 
        with tf.variable_scope('read_head_addressing'):

            # Cosine Similarity

            k = tf.expand_dims(k, axis=2)
            inner_product = tf.matmul(prev_M, k)
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True)) #cosine similiraity
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
            norm_product = M_norm * k_norm
            K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (17)

            # Calculating w^c

            K_exp = tf.exp(K)
            w = K_exp / tf.reduce_sum(K_exp, axis=1, keep_dims=True)                # eq (18)

            return w  #normalized wights 

    def write_head_addressing(self, sig_alpha, prev_w_r, prev_w_lu):
        with tf.variable_scope('write_head_addressing'):

            # Write to (1) the place that was read in t-1 (2) the place that was least used in t-1

            return sig_alpha * prev_w_r + (1. - sig_alpha) * prev_w_lu              # eq (22)

    def least_used(self, w_u): #this is the eq:21 this is to assign the leased used waight position to 1  and others to zero
        
        _, indices = tf.nn.top_k(w_u, k=self.memory_size)  #this will output 128 indices last indice is the lowest value #128 values in decending order
        w_lu = tf.reduce_sum(tf.one_hot(indices[:, -self.head_num:], depth=self.memory_size), axis=1) #making the vector one hot leased used weight to 1 and others to zero
  
        return indices, w_lu 

    def zero_state(self, batch_size, dtype):
        one_hot_weight_vector = np.zeros([batch_size, self.memory_size])
        one_hot_weight_vector[..., 0] = 1
        one_hot_weight_vector = tf.constant(one_hot_weight_vector, dtype=tf.float32)

        with tf.variable_scope('init', reuse=self.reuse):
            state = {
                'controller_state': self.controller.zero_state(batch_size, dtype),    #we take the hidden state of the 
                'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])  #List of read vectors from each read head
                                     for _ in range(self.head_num)],
                'w_r_list': [one_hot_weight_vector for _ in range(self.head_num)], #this is for read weight list
                'w_u': one_hot_weight_vector,  #the  is equal to number of memmory locations. This is like the usage one
                'M': tf.constant(np.ones([batch_size, self.memory_size, self.memory_vector_dim]) * 1e-6, dtype=tf.float32) #memory matrix  128 * 20
            }
            return state