# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

class ALTRec():
    def __init__(self, args):
        self.p_dims = args.p_dims
        if args.q_dims is None:
            self.q_dims = args.p_dims[::-1]
        else:
            assert args.q_dims[0] == args.p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert args.q_dims[-1] == args.p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = args.q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        self.lam_d = args.lam_d
        self.lam_g = args.lam_g
        self.lr_g = args.lr_g
        self.lr_d = args.lr_d   
        self.adv_coeff = args.adv_coeff
        self.construct_placeholders()

    def construct_weights(self):
        self.weights_gen = []
        self.biases_gen = []
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_gen_{}to{}".format(i, i + 1)
            bias_key = "bias_gen_{}".format(i + 1)
            self.weights_gen.append(tf.get_variable(name=weight_key, shape=[d_in, d_out], initializer=tf.contrib.layers.xavier_initializer()))
            self.biases_gen.append(tf.get_variable(name=bias_key, shape=[d_out], initializer=tf.truncated_normal_initializer()))
        self.params_gen = self.weights_gen + self.biases_gen

        dims = [self.p_dims[-1]*2, 200]
        self.weights_dis, self.biases_dis = [], []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            weight_key = "weight_dis_{}to{}".format(i, i+1)
            bias_key = "bias_dis_{}".format(i+1)
            self.weights_dis.append(tf.get_variable(name=weight_key, shape=[d_in, d_out], initializer=tf.contrib.layers.xavier_initializer()))
            self.biases_dis.append(tf.get_variable(name=bias_key, shape=[d_out], initializer=tf.truncated_normal_initializer(stddev=0.001)))
        self.params_dis = self.weights_dis + self.biases_dis

    def construct_placeholders(self):
        self.input_ph1 = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.input_ph2 = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.confidence = tf.placeholder_with_default(1., shape=None)

    def forward_pass(self, inp):
        h = inp
        for i, (w, b) in enumerate(zip(self.weights_gen, self.biases_gen)):
            h = tf.matmul(h, w) + b
            h = tf.nn.sigmoid(h)
        return h

    def discriminate(self, inp):
        h = inp
        for i, (w, b) in enumerate(zip(self.weights_dis, self.biases_dis)):
            h = tf.matmul(h, w) + b
            h = tf.nn.sigmoid(h)    
        return h

    def build_graph(self):
        self.construct_weights()
        self.logits1 = self.forward_pass(self.input_ph1)
        self.logits2 = self.forward_pass(self.input_ph2)

        self.logits2 = tf.stop_gradient(self.logits2)
        rep_t = self.discriminate(inp=tf.concat([self.input_ph1, self.input_ph2], axis=1))
        rep_f = self.discriminate(inp=tf.concat([self.input_ph1*self.logits1, self.input_ph2*self.logits2], axis=1))

        uniform_noise = tf.random_uniform(shape=[tf.shape(self.input_ph1)[0], 1], minval=0., maxval=1.)
        interpolates = uniform_noise * tf.concat([self.input_ph1, self.input_ph2], axis=1) + (1. - uniform_noise) * tf.concat([self.input_ph1 * self.logits1, self.input_ph2 * self.logits2], axis=1) + 1e-2

        rep_interpolate = self.discriminate(inp=interpolates)
        self.interpolates = tf.reduce_sum(interpolates)
        self.rep_interpolate = rep_interpolate
        gradients = tf.gradients(rep_interpolate, [interpolates])[0]
        penalty = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
        penalty = tf.reduce_mean(tf.square(penalty - 1.))
        self.penalty = self.lam_d * penalty
        self.gradients = gradients        
        
        # l2 regularization.
        reg_g = l2_regularizer(self.lam_g)
        reg_var_g = apply_regularization(reg_g, self.params_gen)

        self.loss_sim_d = -tf.reduce_mean(self.confidence * tf.reduce_sum((rep_t - rep_f)**2, axis=1)) #square loss
        self.loss_sim_g = tf.reduce_mean(self.confidence * tf.reduce_sum((rep_t - rep_f)**2, axis=1)) #square loss
        self.loss_dis = self.loss_sim_d + self.lam_d * penalty
                
        
        self.loss_rec = tf.reduce_mean(tf.reduce_sum(-self.input_ph1 * tf.log(self.logits1 + 1e-8) - (1. - self.input_ph1) * tf.log(1. - self.logits1 + 1e-8), axis=1))
        self.loss_gen = self.loss_rec + self.adv_coeff * self.loss_sim_g + 2. * reg_var_g

        self.op_gen = tf.train.AdamOptimizer(self.lr_g).minimize(loss=self.loss_gen, var_list=self.params_gen)
        self.op_dis = tf.train.AdamOptimizer(self.lr_d).minimize(loss=self.loss_dis, var_list=self.params_dis)
