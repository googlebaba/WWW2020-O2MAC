import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, model, preds_fuze, preds, p, labels,  numView,\
         pos_weights, fea_pos_weights, norm):
    #labels 为adj_origin
        preds_sub = preds
        labels_sub = labels
        embed = model.embeddings

        self.cost = 0
        self.cost_list = []
        all_variables = tf.trainable_variables()
        
        self.l2_loss = 0
        for var in all_variables:
            self.l2_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for v in range(numView):
            self.cost += norm[v] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.reshape(preds_fuze[v], [-1]), targets=tf.reshape(labels_sub[v], [-1]), pos_weight=pos_weights[v]))
            cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.reshape(preds_sub[v], [-1]), targets=tf.reshape(labels_sub[v], [-1]), pos_weight=pos_weights[v]))
            self.cost_list.append(cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, 
                                           beta1=0.9, name='adam')  # Adam Optimizer
        self.cost = self.cost + self.l2_loss
        q = model.cluster_layer_q
        kl_loss = tf.reduce_sum(p * tf.log(p/q))
        self.cost_kl = self.cost + FLAGS.kl_decay* kl_loss
     
        self.opt_op = self.optimizer.minimize(self.cost)
        self.opt_op_list = []
        for v in range(numView):
            opt_op1 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost_list[v])
            self.opt_op_list.append(opt_op1)

        self.opt_op_kl = self.optimizer.minimize(self.cost_kl)
               
    
    def target_distribution(self, q):
        weight = tf.pow(q, 2) / tf.reduce_sum(q, axis=0)
        return tf.transpose((tf.transpose(weight)/tf.reduce_sum(weight, axis=1)))




class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]


        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var)#minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss,
                                                                                                var_list=en_var)

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
