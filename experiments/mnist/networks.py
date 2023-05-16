# coding=utf-8
from core.utils.utils import dotdict as dd


n_classes = 10
bignet = [dd(name='conv1', type='conv', n_filters=8),
          # dotdict(name='avg_pool1', type='avg_pool'),
          dd(name='conv2', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5))),
          dd(name='avg_pool2', type='avg_pool'),
          dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
          dd(name='avg_pool3', type='avg_pool'),
          dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
          # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
          dd(name='dense1', type='dense', outdim=n_classes,
             factors=dd(coeff_prior=dd(sigma=1., sigma_test_mult=1.),
                        weight_prior=dd(sigma=1., sigma_test_mult=1.))),
          dd(name='softmax1', type='softmax_class_obs')]
bignet_2k3 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
             dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,),
             dd(name='softmax1', type='softmax_class_obs')]
bignet_k5 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, nonlin='none'))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,),
             dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_nonlin_last = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,),
             dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_last_ff = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, feedforward=True))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,),
             dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_maxpool = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                     dd(name='max_pool2', type='max_pool'),
                     dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
                     dd(name='max_pool3', type='max_pool'),
                     dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, nonlin='none'))),
                     # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                     dd(name='dense1', type='dense', outdim=n_classes,),
                     dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_binary_class = \
    [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,),
             dd(name='binary1', type='binary_class_obs')]

bignet_k5_twofiltlast = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=2, factors=dd(recon=dd(sigma=0.4))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,),
             dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_16 = [dd(name='conv1', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes),
             dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_ff = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5, feedforward=True))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3, feedforward=True))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, feedforward=True))),
             # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,
                factors=dd(coeff_prior=dd(sigma=1., sigma_test_mult=1.),
                           weight_prior=dd(sigma=1., sigma_test_mult=1.))),
             dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_extra_conv = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             dd(name='avg_pool2', type='avg_pool'),
             dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
             dd(name='avg_pool3', type='avg_pool'),
             dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
             dd(name='conv5', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
             dd(name='dense1', type='dense', outdim=n_classes,
                factors=dd(coeff_prior=dd(sigma=1., sigma_test_mult=1.),
                           weight_prior=dd(sigma=1., sigma_test_mult=1.))),
             dd(name='softmax1', type='softmax_class_obs')]


deepnet = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='conv2', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='conv3', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='conv4', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='avg_pool1', type='avg_pool'),
           dd(name='conv5', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='conv6', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='conv7', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           dd(name='conv8', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=3))),
           # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
           dd(name='dense1', type='dense', outdim=n_classes,
              factors=dd(coeff_prior=dd(sigma=1., sigma_test_mult=1.),
                         weight_prior=dd(sigma=1., sigma_test_mult=1.))),
           dd(name='softmax1', type='softmax_class_obs')]


bignet_k5_32 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                dd(name='avg_pool2', type='avg_pool', factors=dd(pool=dd(ksize=2))),
                dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
                dd(name='avg_pool3', type='avg_pool', factors=dd(pool=dd(ksize=2))),
                dd(name='conv4', type='conv', n_filters=32, factors=dd(recon=dd(sigma=0.4))),
                dd(name='dense1', type='dense', outdim=n_classes),
                dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_16first_32 = [dd(name='conv1', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                dd(name='avg_pool2', type='avg_pool', factors=dd(pool=dd(ksize=2))),
                dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
                dd(name='avg_pool3', type='avg_pool', factors=dd(pool=dd(ksize=2))),
                dd(name='conv4', type='conv', n_filters=32, factors=dd(recon=dd(sigma=0.4))),
                dd(name='dense1', type='dense', outdim=n_classes),
                dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_32_32 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                dd(name='avg_pool2', type='avg_pool'),
                dd(name='conv3', type='conv', n_filters=32, factors=dd(recon=dd(sigma=0.3))),
                dd(name='avg_pool3', type='avg_pool'),
                dd(name='conv4', type='conv', n_filters=32, factors=dd(recon=dd(sigma=0.4))),
                # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                dd(name='dense1', type='dense', outdim=n_classes),
                dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_64 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                dd(name='avg_pool2', type='avg_pool', factors=dd(pool=dd(ksize=2))),
                dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
                dd(name='avg_pool3', type='avg_pool', factors=dd(pool=dd(ksize=2))),
                dd(name='conv4', type='conv', n_filters=64, factors=dd(recon=dd(sigma=0.4))),
                dd(name='dense1', type='dense', outdim=n_classes),
                dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_32_twomidconv = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                dd(name='avg_pool2', type='avg_pool'),
                dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
                dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
                dd(name='avg_pool3', type='avg_pool'),
                dd(name='conv5', type='conv', n_filters=32, factors=dd(recon=dd(sigma=0.4))),
                # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                dd(name='dense1', type='dense', outdim=n_classes),
                dd(name='softmax1', type='softmax_class_obs')]

bignet_2k5 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
              dd(name='conv2', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
              dd(name='avg_pool1', type='avg_pool', factors=dd(avg_pool=dd(ksize=2))),
              dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
              dd(name='avg_pool3', type='avg_pool', factors=dd(avg_pool=dd(ksize=2))),
              dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
              # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
              dd(name='dense1', type='dense', outdim=n_classes,
                 factors=dd(coeff_prior=dd(sigma=1., sigma_test_mult=1.),
                            weight_prior=dd(sigma=1., sigma_test_mult=1.))),
              dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_extra_dense = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
              dd(name='avg_pool1', type='avg_pool', factors=dd(avg_pool=dd(ksize=2))),
              dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
              dd(name='avg_pool3', type='avg_pool', factors=dd(avg_pool=dd(ksize=2))),
              dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
              # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
              dd(name='dense1', type='dense', outdim=n_classes * 2, factors=dd(dense=dd(nonlin='tanh'))),
              dd(name='dense2', type='dense', outdim=n_classes),
              dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_nolastnonlin = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
              dd(name='avg_pool1', type='avg_pool', factors=dd(avg_pool=dd(ksize=2))),
              dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3))),
              dd(name='avg_pool3', type='avg_pool', factors=dd(avg_pool=dd(ksize=2))),
              dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, nonlin='none'))),
              # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
              dd(name='dense1', type='dense', outdim=n_classes),
              dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_scaled_priors = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                           dd(name='avg_pool2', type='avg_pool'),
                           dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3), coeff_prior=dd(sigma=1.))),
                           dd(name='avg_pool3', type='avg_pool'),
                           dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4),
                                                                                  coeff_prior=dd(sigma=2.))),
                           # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                           dd(name='dense1', type='dense', outdim=n_classes,
                              factors=dd(coeff_prior=dd(sigma=4., sigma_test_mult=1.))),
                           dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_scaled_priors_nolastnonlin = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                           dd(name='avg_pool2', type='avg_pool'),
                           dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3), coeff_prior=dd(sigma=1.))),
                           dd(name='avg_pool3', type='avg_pool'),
                           dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, nonlin='none'),
                                                                                  coeff_prior=dd(sigma=2.),
                                                                                  weight_prior=dd(sigma=0.5))),
                           # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                           dd(name='dense1', type='dense', outdim=n_classes,
                              factors=dd(coeff_prior=dd(sigma=4., sigma_test_mult=1.))),
                           dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_scaled2_priors_nolastnonlin = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                           dd(name='avg_pool2', type='avg_pool'),
                           dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3), coeff_prior=dd(sigma=1.))),
                           dd(name='avg_pool3', type='avg_pool'),
                           dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, nonlin='none'),
                                                                                  coeff_prior=dd(sigma=3.),
                                                                                  weight_prior=dd(sigma=0.3))),
                           # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                           dd(name='dense1', type='dense', outdim=n_classes,
                              factors=dd(coeff_prior=dd(sigma=4., sigma_test_mult=1.))),
                           dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_scaled3_priors_nolastnonlin = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                           dd(name='avg_pool2', type='avg_pool'),
                           dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3), coeff_prior=dd(sigma=1.))),
                           dd(name='avg_pool3', type='avg_pool'),
                           dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4, nonlin='none'),
                                                                                  coeff_prior=dd(sigma=3.),
                                                                                  weight_prior=dd(sigma=0.5))),
                           # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                           dd(name='dense1', type='dense', outdim=n_classes,
                              factors=dd(coeff_prior=dd(sigma=4., sigma_test_mult=1.))),
                           dd(name='softmax1', type='softmax_class_obs')]

bignet_k5_k5 = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                dd(name='avg_pool2', type='avg_pool'),
                dd(name='conv3', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.3, ksize=5))),
                dd(name='avg_pool3', type='avg_pool'),
                dd(name='conv4', type='conv', n_filters=16, factors=dd(recon=dd(sigma=0.4))),
                # dd(name='avg_pool4', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                dd(name='dense1', type='dense', outdim=n_classes,
                   factors=dd(coeff_prior=dd(sigma=1., sigma_test_mult=1.),
                              weight_prior=dd(sigma=1., sigma_test_mult=1.))),
                dd(name='softmax1', type='softmax_class_obs')]

vsmallnet = [dd(name='conv1', type='conv', n_filters=8, factors=dd(recon=dd(ksize=5))),
             dd(name='avg_pool1', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='conv2', type='conv', n_filters=8, factors=dd(recon=dd(sigma=0.5, ksize=5))),
             # dd(name='avg_pool2', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
             dd(name='dense1', type='dense', outdim=n_classes,
                factors=dd(weight_prior=dd(sigma=1.), coeff_prior=dd(sigma=1.))),
             dd(name='softmax1', type='softmax_class_obs')]


vsmallnet_4filt = [dd(name='conv1', type='conv', n_filters=4, factors=dd(recon=dd(ksize=5))),
                   dd(name='avg_pool1', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                   dd(name='conv2', type='conv', n_filters=4, factors=dd(recon=dd(sigma=0.5, ksize=5))),
                   # dd(name='avg_pool2', type='avg_pool', factors=dd(avg_pool=dd(ksize=3))),
                   dd(name='dense1', type='dense', outdim=n_classes,
                      factors=dd(weight_prior=dd(sigma=1.), coeff_prior=dd(sigma=1.))),
                   dd(name='softmax1', type='softmax_class_obs')]

single_layer_conv4 = [dd(name='conv1', type='conv', n_filters=4, factors=dd(recon=dd(ksize=3, sigma=0.3)))]

architectures = {'bignet': bignet,
                 'bignet_k5': bignet_k5,
                 'bignet_2k3': bignet_2k3,
                 'bignet_k5_binary': bignet_k5_binary_class,
                 'bignet_k5_extra_dense': bignet_k5_extra_dense,
                 'bignet_k5_twofiltlast': bignet_k5_twofiltlast,
                 'bignet_k5_16': bignet_k5_16,
                 'bignet_k5_64': bignet_k5_64,
                 'bignet_k5_ff': bignet_k5_ff,
                 'bignet_k5_last_ff': bignet_k5_last_ff,
                 'bignet_k5_nonlin_last': bignet_k5_nonlin_last,
                 'bignet_k5_extra_conv': bignet_k5_extra_conv,
                 'bignet_k5_32': bignet_k5_32,
                 'bignet_k5_16first_32': bignet_k5_16first_32,
                 'bignet_k5_32_32': bignet_k5_32_32,
                 'bignet_k5_32_twomidconv': bignet_k5_32_twomidconv,
                 'bignet_k5_scaled': bignet_k5_scaled_priors,
                 'bignet_k5_maxpool': bignet_k5_maxpool,
                 'bignet_k5_scaled_nolastnonlin': bignet_k5_scaled_priors_nolastnonlin,
                 'bignet_k5_scaled2_nolastnonlin': bignet_k5_scaled2_priors_nolastnonlin,
                 'bignet_k5_scaled3_nolastnonlin': bignet_k5_scaled3_priors_nolastnonlin,
                 'bignet_k5_nolastnonlin': bignet_k5_nolastnonlin,
                 'bignet_k5_k5': bignet_k5_k5,
                 'bignet_2k5': bignet_2k5,
                 'deepnet': deepnet,
                 'vsmallnet': vsmallnet,
                 'vsmallnet_4filt': vsmallnet_4filt,
                 'single_layer_conv_4': single_layer_conv4
                 }





def get_network(net_name, conv_sigmas=None, dense_coeff_prior_sigma=None, dense_weight_prior_sigma=None, conv_coeff_prior_sigmas=None, conv_coeff_prior_N_robs=None):
    net = architectures[net_name]

    if conv_sigmas is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'recon' not in layconf.factors:
                    layconf.factors.recon = dd()
                layconf.factors.recon.sigma = conv_sigmas[conv_lay_id]
                conv_lay_id += 1

        assert conv_lay_id == len(conv_sigmas), \
            f"Length of recon factor sigmas {conv_sigmas} " \
            f"does not match number of conv recon layers in network {net_name}: " \
            f"{len(conv_sigmas)} != {conv_lay_id}"

    if conv_coeff_prior_sigmas is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'coeff_prior' not in layconf.factors:
                    layconf.factors.coeff_prior = dd()
                layconf.factors.coeff_prior.sigma = conv_coeff_prior_sigmas[conv_lay_id]
                conv_lay_id += 1

    if conv_coeff_prior_N_robs is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'coeff_prior' not in layconf.factors:
                    layconf.factors.coeff_prior = dd()
                layconf.factors.coeff_prior.N_rob = conv_coeff_prior_N_robs[conv_lay_id]
                conv_lay_id += 1

    dense_to_update = (('coeff_prior', dense_coeff_prior_sigma),) if dense_coeff_prior_sigma is not None else ()
    dense_to_update += (('weight_prior', dense_weight_prior_sigma),) if dense_weight_prior_sigma is not None else ()
    if len(dense_to_update):
        for layconf in net:
            if layconf.type == 'dense':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                for fact_type, sigma in dense_to_update:
                    if fact_type not in layconf.factors:
                        setattr(layconf.factors, fact_type, dd())
                    getattr(layconf.factors, fact_type).sigma = sigma

    return net
