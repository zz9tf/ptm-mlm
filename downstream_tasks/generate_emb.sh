short_model_name="c600" # c6B, 2-650, 2-15b
model_name="EvolutionaryScale/esmc-600m-2024-12" # 
layer_index=30

task run "p site $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/p_site_prediction/generate_emb_p_site.sh $model_name $layer_index"
task run "nha site $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/NHA_site_prediction/generate_emb_nhas.sh $model_name $layer_index"
task run "ppi $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction/generate_emb_ppi.sh $model_name $layer_index"