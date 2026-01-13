# short_model_name="2-650" # c6B, 2-650, 2-15b
# model_name="facebook/esm2_t33_650M_UR50D" # 
# layer_index=32

short_model_name="c600" # c6B, 2-650, 2-15b
model_name="EvolutionaryScale/esmc-600m-2024-12" # 
layer_index="None"

task run "p site $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/p_site_prediction/generate_emb.sh $model_name $layer_index 1"
task run "nha site $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/NHA_site_prediction/generate_emb.sh $model_name $layer_index 2"
task run "ppi $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction/generate_emb.sh $model_name $layer_index 4"