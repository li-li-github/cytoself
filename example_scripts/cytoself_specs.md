

# CytoselfFullTrainer model_args

 - `input_shape`: (3, 100, 100) 
    - The shape of the input image conisting of 3 channels:
        - Labeled Protein of Interest
        - Nuclear fudicial channel
        - Nuclear distance transform
 - `output_shape`: (3, 100, 100) 
    - The shape of the output reconstructed image, in most cases will be the same as the input_shape
 - `emb_shapes`: (emb_fc1_shape, emb_fc2_shape): ((25, 25), (4, 4))
    - emb_fc1_shape: (25, 25, 64)
        - The local embedding corresponding to the output of VQ1 with shape (25, 25, 64)
    - emb_fc2_shape: (4, 4, 576)
        - The global embedding, corresponding to the output of VQ2 with shape (4, 4, 576)
    - For both embedding shapes only the first 2 dims are adjustable, the 3rd is specified by the model architecture
 - `fc_output_idx`: "all"
    - Controls which VQ layer is used for the auxilliary task of protein classification. Set to "all" because the outputs from both VQ1 and VQ2 are used for classification.
    - Options:
        - [1] only VQ1 (local embeddings) are used for classification
        - [2] only VQ2 (global embeddings) are used for classification
        - "all" both VQ1 and VQ2 used
 - `vq_args`: {"num_embeddings": 2048, "embedding_dim": 64}
    - the specifications of the VQ-VAE codebook, which consists of 2048 codes (num_embeddings=2048), 
    with each code consisting of 64 features or dimensions (embedding_dim=64)
 - `fc_input_type`: "vqvec"
    - defines the input to the fully connected layers
    - Options:
        - "vqvec": the quantized vector output from the codebook, the most informative option
        - "vqind": the quantized index into the codebook
        - "vqindhist": the histogram count of how often each codebook index is used, acts as a low dimensional summary of the embedding