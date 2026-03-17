# nohup bash ./train_w_challenge_multi_lang_plms_subtaskA.sh > train_w_challenge_multi_lang_plms_subtaskA_results.log 2>&1 &

### mbert
for config in "../config/config-mbert_base.txt" "../config/config-mt5_base.txt" "../config/config-xlmroberta_base.txt"

do
    echo "start training config ${config}......"

    for single_dataset in "cstance" "ezstance"
    do
        echo "start training single_dataset ${single_dataset}......"
        for target_type in "claim_challenge" #"mixed_challenge"
        do
            echo "start training target_type ${target_type}......"
            ###################################################################################################################
            ###                          train on claim                                    
            ###################################################################################################################
            train_data=/data/czhao43/bilingual/data/${single_dataset}/subtaskA_dataset_split/${target_type}/raw_train_all_onecol.csv
            dev_data=/data/czhao43/bilingual/data/${single_dataset}/subtaskA_dataset_split/${target_type}/raw_val_all_onecol.csv
            test_data=/data/czhao43/bilingual/data/${single_dataset}/subtaskA_dataset_split/${target_type}/raw_test_all_onecol.csv

            if [ "${single_dataset}" == "cstance" ]; then
                train_mode="train_zh_test_zh"
            elif [ "${single_dataset}" == "ezstance" ]; then 
                train_mode="train_en_test_en"
            elif [ "${single_dataset}" == "bilingual" ]; then 
                train_mode="train_bi_test_bi"
            fi

            for seed in {0..3}
            do
                echo "start training seed ${seed}......"

                python train_model.py -prompt_index 0 -mode ${train_mode} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} \
                                      -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100


                ### evaluate easy (original) version of dataset
                echo "start testing original eval_dataset ${eval_dataset}......"
                for eval_dataset in "cstance" "ezstance" "bilingual"
                do
                    echo "start testing eval_dataset ${eval_dataset}......"
                    for target_type in "noun_phrase" "claim" "mixed"
                    do
                        echo "start testing target_type ${target_type}......"
                        test_data_noun_phrase=/data/czhao43/bilingual/data/${eval_dataset}/subtaskA_dataset_split/${target_type}/raw_test_all_onecol.csv
                        
                        if [ "${eval_dataset}" == "cstance" ]; then
                            test_mode="train_zh_test_zh"
                        elif [ "${eval_dataset}" == "ezstance" ]; then 
                            test_mode="train_en_test_en"
                        elif [ "${eval_dataset}" == "bilingual" ]; then 
                            test_mode="train_bi_test_bi"
                        fi

                        python eval_model.py -mode ${test_mode} -name "${eval_dataset} ${target_type}" -prompt_index 0 -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data_noun_phrase}\
                                          -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
                    done
                done




                ### evaluate challenging (rephrased) version of dataset
                echo "start testing challenging eval_dataset ${eval_dataset}......"
                for eval_dataset in "cstance" "ezstance" "bilingual"
                do
                    echo "start testing eval_dataset ${eval_dataset}......"
                    for target_type in "claim_challenge" "mixed_challenge"
                    do
                        ### ori test set and challenge test set
                        echo "start testing target_type ${target_type}......"
                        test_data_noun_phrase=/data/czhao43/bilingual/data/${eval_dataset}/subtaskA_dataset_split/${target_type}/raw_test_all_onecol.csv
                        
                        if [ "${eval_dataset}" == "cstance" ]; then
                            test_mode="train_zh_test_zh"
                        elif [ "${eval_dataset}" == "ezstance" ]; then 
                            test_mode="train_en_test_en"
                        elif [ "${eval_dataset}" == "bilingual" ]; then 
                            test_mode="train_bi_test_bi"
                        fi

                        python eval_model.py -mode ${test_mode} -name "${eval_dataset} ${target_type} ori and challenging" -prompt_index 0 -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data_noun_phrase}\
                                          -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
                        
                        ### challenge data only
                        echo "start testing target_type ${target_type}......"
                        test_data_noun_phrase=/data/czhao43/bilingual/data/${eval_dataset}/subtaskA_dataset_split/${target_type}/raw_test_all_onecol_challengeOnly.csv
                        
                        python eval_model.py -mode ${test_mode} -name "${eval_dataset} ${target_type} challengingOnly" -prompt_index 0 -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data_noun_phrase}\
                                          -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
                        

                        ### ori version of challenging data only
                        echo "start testing target_type ${target_type}......"
                        test_data_noun_phrase=/data/czhao43/bilingual/data/${eval_dataset}/subtaskA_dataset_split/${target_type}/raw_test_all_onecol_ori.csv
                        
                        python eval_model.py -mode ${test_mode} -name "${eval_dataset} ${target_type} easy version of challengingOnly" -prompt_index 0 -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data_noun_phrase}\
                                          -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
                    done
                done
            done
        done
    done

done
