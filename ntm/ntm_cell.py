import tensorflow as tf
import numpy as np

class NTMCell():  #this is the class for the NTM cell . Same as LSTM cell but with external memory as in the paper 
    def __init__(self, rnn_size, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 addressing_mode='content_and_loaction', shift_range=1, reuse=False, output_dim=None):
        self.rnn_size = rnn_size #hidden dimentions of the RNN cell 
        self.memory_size = memory_size  #internal size of a memory vector 
        self.memory_vector_dim = memory_vector_dim #length of the memory vector
        self.read_head_num = read_head_num  #how many heads 
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.controller = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)  #this is the controller network 
        self.step = 0
        self.output_dim = output_dim #
        self.shift_range = shift_range #this is for the location based adressing to do the circuar convolution

    def __call__(self, x, prev_state): #we put the data in the previous state and first the input 

#so the whole system is to make the NTM keep remebering what it did. First it will add the input to the previous read memory . 
#it's like we get something out that . how the last seen case intracted with the memory. 

        prev_read_vector_list = prev_state['read_vector_list']      # read vector in Sec 3.1 (the content that is
                                                                    # read out, length = memory_vector_dim)
        prev_controller_state = prev_state['controller_state']      # state of controller (LSTM hidden state)
                                                                    #can take as the controller state RNN 
                                                         
        # x + prev_read_vector -> controller (RNN) -> controller_output

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1) #here add the input 9 bit vector to read vector from the previous memorty locations 

        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_controller_state) #send through the lstm . Controller output contains (batchsize * 128(rnn hidden num))


        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M, Sec 3.1)
        #                       -> beta (positive scalar, key strength, Sec 3.1)                -> w^c
        #                       -> g (scalar in (0, 1), blend between w_prev and w^c, Sec 3.2)  -> w^g
        #                       -> s (dim = shift_range * 2 + 1, shift weighting, Sec 3.2)      -> w^~
        #                            (not memory_size, that's too wide)
        #                       -> gamma (scalar (>= 1), sharpen the final result, Sec 3.2)     -> w    * num_heads
        # controller_output     -> erase, add vector (dim = memory_vector_dim, \in (0, 1), Sec 3.2)     * write_head_num

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1 #here memory vec dim is equal to the kt . That is the veecto we are producing doing content based matching 
  
        num_heads = self.read_head_num + self.write_head_num #two heads should predict these things 
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num

        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            o2p_w = tf.get_variable('o2p_w', [controller_output.get_shape()[1], total_parameter_num], #prediction layer weight
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            o2p_b = tf.get_variable('o2p_b', [total_parameter_num],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            parameters = tf.nn.xw_plus_b(controller_output, o2p_w, o2p_b) #computer the output multiply by weights and add bias
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1) #spliting parameters 

        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1) #these are earase and add when used in reading and writing 
        #above thing is the erase content before finaly updatig the memory vector. this is element vice operation so this equal to the dimentions of the one memroy vector

        # k, beta, g, s, gamma -> w

        prev_w_list = prev_state['w_list']  # vector of weightings (blurred address) initialized softmax  scores for each memory location we use this for the interpolation 
        prev_M = prev_state['M'] #previous memory matrix 
        w_list = [] 
        p_list = []
  
        for i, head_parameter in enumerate(head_parameter_list): #read right head weights  #this is interating only two times one for the read and one for the write params 
#here the same read and wright weight heads are distributed between read and write vector parts 
            # Some functions to constrain the result in specific range
            # exp(x)                -> x > 0
            # sigmoid(x)            -> x \in (0, 1)
            # softmax(x)            -> sum_i x_i = 1
            # log(exp(x) + 1) + 1   -> x > 1
#if there are two read head s and two write heads this will ierate for 4 times 

            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim]) #getting the K values  inoder to calculate the content based similarity . This is for the cosine simelaritY

            beta = tf.sigmoid(head_parameter[:, self.memory_vector_dim]) * 10    # do not use exp, it will explode! this use in content based attention to attenuate the normalizer
          
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])  #this is the interpolation gate . Tis use to update current wcontent based weights with previous ontest based wieghts 
        
            s = tf.nn.softmax(       #these parameters are for the shift and direct access 
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)] #this is to deside whether and by how much we need to rorate the weifjts
            )
            #if the shift is one u can be at the same place of move to left or right so there should be 3 parameters they do circular convolution 
            gamma = tf.log(tf.exp(head_parameter[:, -1]) + 1) + 1      #this is for the shaprnenning 
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])   # Figure 2 This complte the adressing and update the weights 
            w_list.append(w)  #for each read and write head we need this 
            p_list.append({'k': k, 'beta': beta, 'g': g, 's': s, 'gamma': gamma})    #ths is the parameter list 

        # Reading (Sec 3.1)
   
        read_w_list = w_list[:self.read_head_num]      #these are the final reading vector list after all content and location based adressing 
        read_vector_list = []
        for i in range(self.read_head_num):#we have one read head #if we have two heads we get two read vector which we will be interpolate with the x when putting the input to the lstm controller
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1) #multiply the read vector weights with the previus mememory this is the output of the ntm cell
            read_vector_list.append(read_vector)  #only one read vector 

        # Writing (Sec 3.2)

        write_w_list = w_list[self.read_head_num:] #get the write vectors same as above if we have  final write head paraeters 
        M = prev_M
        for i in range(self.write_head_num): #updating the memory vector  #only one head 
            w = tf.expand_dims(write_w_list[i], axis=2)  #here the writing weight also goes throug locations based and everything as the reading weights . 
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1) #get the erase vector inbetween 0 and 1
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1) #write vector between -1 and + 1 
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector) #update the memory vector 

        # controller_output -> NTM output

        if not self.output_dim:
            output_dim = x.get_shape()[1]  #input shape at one time step 
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):    
            o2o_w = tf.get_variable('o2o_w', [controller_output.get_shape()[1], output_dim],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            o2o_b = tf.get_variable('o2o_b', [output_dim],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            NTM_output = tf.nn.xw_plus_b(controller_output, o2o_w, o2o_b)

        state = {
            'controller_state': controller_state, #hidden state 
            'read_vector_list': read_vector_list, #this has the sum of the read vector list in the section 3.1 
            'w_list': w_list, #this has the normalized weight list 
            'p_list': p_list, #paramter list got updated 
            'M': M  #memory vector got updated 
        }

        self.step += 1
     
        return NTM_output, state

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = tf.expand_dims(k, axis=2)  #this is to get the values of the vector 
        inner_product = tf.matmul(prev_M, k)  #get unner prodct with the each memory locations there can be 128 memory locations 
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))           #getting the cosine similarity 
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8)) #get the cosine similarity                # eq (6)

        # Calculating w^c

        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K) #amplification or atanouation of the cosign similarity with the beta
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  # eq (5) get the normalize things 

        if self.addressing_mode == 'content':         # Only focus on content #if adressing mode is only content 
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = tf.expand_dims(g, axis=1)  #get the interpolation gate 
        w_g = g * w_c + (1 - g) * prev_w               # eq (7) element wise multiplication  this is like remebeing or keeping the old weigjts with gate 

        s = tf.concat([s[:, :self.shift_range + 1],  #this is circular convolution matrix 
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))                 #again we normalize
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)        # eq (9)

        return w #getting the location and content based adressing weight for each memory location 

    def zero_state(self, batch_size, dtype):
        def expand(x, dim, N): #N is the batch size
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        with tf.variable_scope('init', reuse=self.reuse):

            state = {            #state is a tupple 
                # 'controller_state': self.controller.zero_state(batch_size, dtype),
                # 'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])
                #                      for _ in range(self.read_head_num)],
                # 'w_list': [tf.zeros([batch_size, self.memory_size])
                #            for _ in range(self.read_head_num + self.write_head_num)],
                # 'M': tf.zeros([batch_size, self.memory_size, self.memory_vector_dim])
                'controller_state': expand(tf.tanh(tf.get_variable('init_state', self.rnn_size,  #this is the hidden state of the bidierctional lstm cell  (batch_size * hidden state)
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size),
                'read_vector_list': [expand(tf.nn.softmax(tf.get_variable('init_r_%d' % i, [self.memory_vector_dim], #this is for the read vector weights (batch_size *length of a memory loca)
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size)         #this is the sum of the weights mutp
                           for i in range(self.read_head_num)],
                'w_list': [expand(tf.nn.softmax(tf.get_variable('init_w_%d' % i, [self.memory_size],   #This is the weightings for each location so 128 weights 
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size) if self.addressing_mode == 'content_and_loaction'
                           else tf.zeros([batch_size, self.memory_size])
                           for i in range(self.read_head_num + self.write_head_num)],
                'M': expand(tf.tanh(tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],  #(batch_sie *size of the memory *number of memory locations)
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),     #size of the memory is 20 , memory locations is 128
                                  dim=0, N=batch_size)
            }
          
            return state