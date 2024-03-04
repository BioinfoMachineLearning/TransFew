cc = {
    'esm2_t48' : {
        'lr': 0.0001, #0.0001, #0.00001, #0.0001
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'msa_1b' : {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'interpro': {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'full' : {
        'lr': {'x': 0.0001, 
               'gcn': 0.0001, #0.00005, #0.0001, #0.00005, #0.00003, #0.0001,#0.00005,
               'biobert': 0.0001,
               'linear': 0.0001,
               },
        'batch_size': 256, #256, #500, #64
        'epochs': 150,
        'weight_decay': 0.0001, #5e-4, #1e-2, #1e-2, #5e-4
        'lr_scheduler': 50,
        'weight_factor': 3
    },
    'label_ae' : {
        'lr': 0.001,
        'weight_decay': 1e-2,
        # 'lr': 0.0001, for GAT cc
    }
}

# rare: 2, full: 3

mf = {
    'esm2_t48' : {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 100,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'msa_1b' : {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'interpro': {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'full' : {
        'lr': {'x': 0.0001, 
               'gcn': 0.0001, #0.0001,
               'biobert': 0.0001,
               'linear': 0.00005,
               },
        'batch_size': 256,
        'epochs': 150,
        'weight_decay': 0.0001, #1e-4, #5e-4
        'lr_scheduler': 50,
        'weight_factor': 3
    },
    'label_ae' : {
        'lr': 0.0001,
        'weight_decay': 1e-2,
    }
}

bp = {
    'esm2_t48' : {
        'lr': 0.0001, #0.00005, #0.0001, #0.00005,
        'batch_size': 500,
        'epochs': 52,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'msa_1b' : {
        'lr': 0.0001,
        'batch_size': 250,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'interpro': {
        'lr': 0.0001,
        'batch_size': 250,
        'epochs': 50,
        'weight_decay': 5e-4,
        'lr_scheduler': 50,
        'weight_factor': 2
    },
    'diamond': {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4
    },
    'string' : {
        'lr': 0.0001,
        'batch_size': 500,
        'epochs': 50,
        'weight_decay': 5e-4
    },
    'full' : {
        'lr': {'x': 0.00005, 
               'gcn': 0.0001,
               'biobert': 0.00005,
               'linear': 0.00001,
               },
        'batch_size': 256,
        'epochs': 150,
        'weight_decay': 1e-4,#1e-2,#5e-4
        'lr_scheduler': 50,
        'weight_factor': 3
    },
    'label_ae' : {
        'lr': 0.0001,
        'weight_decay': 1e-2,
    }

}