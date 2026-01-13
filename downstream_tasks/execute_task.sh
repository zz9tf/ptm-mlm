# short_model_name="c600" # c6B, 2-650, 2-15b
# adaptor_checkpoint="functional"
# model_name="EvolutionaryScale_esmc-600m-2024-12" # 
# layer_index=30

# task run "p site $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/p_site_prediction/execute_task.sh $adaptor_checkpoint $model_name $layer_index 2"
# task run "nha site $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/NHA_site_prediction/execute_task.sh $adaptor_checkpoint $model_name $layer_index 4"
# task run "ppi $short_model_name $layer_index" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction/execute_task.sh $adaptor_checkpoint $model_name $layer_index 5"

task run "ppi c600" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction/execute_task.sh None EvolutionaryScale_esmc-600m-2024-12 None 4"
# task run "ppi 2-650" "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction/execute_task.sh None esm2_t33_650M_UR50D 32 5"