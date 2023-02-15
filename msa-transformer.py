from biotransformers import BioTransformers

msa_folder = "D:/Workspace/python-3/TFUN/data/msa"
bio_trans = BioTransformers("esm_msa1b_t12_100M_UR50S", num_gpus=0)
msa_embeddings = bio_trans.compute_embeddings(sequences=msa_folder, pool_mode=("cls", "mean"), n_seqs_msa=128)

print(msa_embeddings["cls"])

print(msa_embeddings["mean"])