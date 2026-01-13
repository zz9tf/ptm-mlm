task run "generate 650 32 combined emb" '/home/zz/zheng/ptm-mlm/main_pipeline/generate_embeddings1.sh'
task run "generate 650 32 mamba emb" '/home/zz/zheng/ptm-mlm/main_pipeline/generate_embeddings2.sh'
task run "generate 650 32 functional emb" '/home/zz/zheng/ptm-mlm/main_pipeline/generate_embeddings3.sh'