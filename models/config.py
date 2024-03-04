biobert = 768
bert = 1024

##### cellular component #####
cc_out_shape = 2957
cc_out_freq_shape = 873 #2957# 873
cc_out_rare_shape = 2083

esm_layers_freq_cc = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (256, cc_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]
esm_layers_rare_cc = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (256, cc_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]

esm_layers_cc = [ # layer 0
                  (5120, 2048, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (2048, 1024, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (1024, cc_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]

msa_layers_cc = [  # 0 input 
      (768, 1024, True, 'gelu', 'batchnorm', 'none', (0, )),
      (1024, 930, True, 'gelu', 'batchnorm', 'none', (1, )),
      (930, cc_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
]

interpro_layers_cc = [
                      (24714, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                      (1024, 930, True, 'gelu', 'batchnorm', 0.2, (1, )),
                      (930, cc_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
                      ]



##### molecular function #####
mf_out_shape = 7224
mf_out_freq_shape = 1183# 7224 #1183
mf_out_rare_shape =  6040

interpro_layers_mf = [
                      (25523, 1400, True, 'gelu', 'batchnorm', 0.2, (0, )),
                      (1400, 1200, True, 'gelu', 'batchnorm', 0.2, (1, )),
                      (1200, mf_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
                      ]

msa_layers_mf = [  # 0 input 
                   (768, 1400, True, 'gelu', 'batchnorm', 'none', (0, )),
                   (1400, 1200, True, 'gelu', 'batchnorm', 'none', (1, )),
                   (1200, mf_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
                ]

esm_layers_mf = [ # layer 0
                  (5120, 2048, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (2048, 1200, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (1200, mf_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]

esm_layers_freq_mf = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (256, mf_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]

esm_layers_rare_mf = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (256, mf_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]



##### biological process #####
bp_out_shape = 21285
bp_out_freq_shape = 6415
bp_out_rare_shape = 6977
bp_out_rare_2_shape = 7892

interpro_layers_bp = [
      (24846, 2048, True, 'gelu', 'batchnorm', 0.2, (0, )),
      (2048, 1200, True, 'gelu', 'batchnorm', 0.2,  (1, )),
      (1200, bp_out_freq_shape, True, 'gelu', 'none', 'none', (2, )),
      ]

esm_layers_bp = [ # layer 0
                  (5120, 2048, True, 'gelu', 'batchnorm', 0.1, (0, )),
                  (2048, 1200, True, 'gelu', 'batchnorm', 0.1, (1, )),
                  (1200, bp_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]

msa_layers_bp = [
                  (768, 2048, True, 'gelu', 'batchnorm', 'none', (0, )),
                  (2048, 1200, True, 'gelu', 'batchnorm', 'none', (1, )),
                  (1200, bp_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]

esm_layers_rare_bp = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.1, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.1, (1, )),
                  (256, bp_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]

esm_layers_rare_2_bp = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.1, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.1, (1, )),
                  (256, bp_out_rare_2_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]

esm_layers_freq_bp = [ # layer 0
                  (5120, 1024, True, 'gelu', 'batchnorm', 0.1, (0, )),
                  (1024, 256, True, 'gelu', 'batchnorm', 0.1, (1, )),
                  (256, bp_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]




####### For ablation studies

##### cellular component #####
interpro_layers_freq_cc = [
                      (24714, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                      (1024, 930, True, 'gelu', 'batchnorm', 0.2, (1, )),
                      (930, cc_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
                      ]
interpro_layers_rare_cc = [
                      (24714, 1024, True, 'gelu', 'batchnorm', 0.2, (0, )),
                      (1024, 930, True, 'gelu', 'batchnorm', 0.2, (1, )),
                      (930, cc_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
                      ] 

msa_layers_freq_cc = [  # 0 input 
      (768, 1024, True, 'gelu', 'batchnorm', 'none', (0, )),
      (1024, 930, True, 'gelu', 'batchnorm', 'none', (1, )),
      (930, cc_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
]

msa_layers_rare_cc = [  # 0 input 
      (768, 1024, True, 'gelu', 'batchnorm', 'none', (0, )),
      (1024, 930, True, 'gelu', 'batchnorm', 'none', (1, )),
      (930, cc_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
] 


##### molecular function #####
msa_layers_freq_mf = [ # layer 0
                  (768, 1400, True, 'gelu', 'batchnorm', 'none', (0, )),
                  (1400, 1200, True, 'gelu', 'batchnorm', 'none', (1, )),
                  (1200, mf_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]
msa_layers_rare_mf = [ # layer 0
                  (768, 1400, True, 'gelu', 'batchnorm', 'none', (0, )),
                  (1400, 1200, True, 'gelu', 'batchnorm', 'none', (1, )),
                  (1200, mf_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
]

interpro_layers_freq_mf = [ # layer 0
                  (25523, 1400, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (1400, 1200, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (1200, mf_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
]

interpro_layers_rare_mf = [ # layer 0
                  (25523, 1400, True, 'gelu', 'batchnorm', 0.2, (0, )),
                  (1400, 1200, True, 'gelu', 'batchnorm', 0.2, (1, )),
                  (1200, mf_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, )),
]



##### biological process #####
interpro_layers_freq_bp = [
      (24846, bp_out_freq_shape + 512, True, 'gelu', 'batchnorm', 0.2, (0, )),
      (bp_out_freq_shape + 512, bp_out_freq_shape + 128, True, 'gelu', 'batchnorm', 0.2,  (1, )),
      (bp_out_freq_shape + 128, bp_out_freq_shape, True, 'gelu', 'none', 'none', (2, )),
      ]
interpro_layers_rare_bp = [
      (24846, bp_out_rare_shape + 512, True, 'gelu', 'batchnorm', 0.2, (0, )),
      (bp_out_rare_shape + 512, bp_out_rare_shape + 128, True, 'gelu', 'batchnorm', 0.2,  (1, )),
      (bp_out_rare_shape + 128, bp_out_rare_shape, True, 'gelu', 'none', 'none', (2, )),
      ]
interpro_layers_rare_2_bp = [
      (24846, bp_out_rare_2_shape + 512, True, 'gelu', 'batchnorm', 0.2, (0, )),
      (bp_out_rare_2_shape + 512, bp_out_rare_2_shape + 128, True, 'gelu', 'batchnorm', 0.2,  (1, )),
      (bp_out_rare_2_shape + 128, bp_out_rare_2_shape, True, 'gelu', 'none', 'none', (2, )),
      ]

msa_layers_freq_bp = [
                  (768, 3200, True, 'gelu', 'batchnorm', 'none', (0, )),
                  (3200, 3104, True, 'gelu', 'batchnorm', 'none', (1, )),
                  (3104, bp_out_freq_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]
msa_layers_rare_bp = [
                  (768, 3200, True, 'gelu', 'batchnorm', 'none', (0, )),
                  (3200, 3104, True, 'gelu', 'batchnorm', 'none', (1, )),
                  (3104, bp_out_rare_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]
msa_layers_rare_2_bp = [
                  (768, 3200, True, 'gelu', 'batchnorm', 'none', (0, )),
                  (3200, 3104, True, 'gelu', 'batchnorm', 'none', (1, )),
                  (3104, bp_out_rare_2_shape, True, 'gelu', 'batchnorm', 'none', (2, ))
               ]