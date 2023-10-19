#! /bin/bash

set -e

outdir="outputs"
mkdir -p ${outdir}

for model in "bigcode/starcoderplus" "gpt-3.5-turbo-16k-0613" "gpt-4-0613"; do
    for base in "folio" "proofwriter"; do
        if [[ ${model} == "bigcode/starcoderplus" ]]; then
            batch_size=5
            max_length=8192 # max model context including prompt
        else # openai max_length refers only to generation
            batch_size=1
            if [[ ${base} == "proofwriter" ]]; then
                max_length=4096 # many more premises to translate
            else
                max_length=1024 
            fi
        fi
        for n in "1" "2" "4" "8"; do
            if [[ ${n} != "8" ]] && [[ ${base} != "folio" || ${model} == "gpt-4-0613" ]]; then
                continue
            fi
            for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                job="cd $(pwd); source activate linc; accelerate launch runner.py"
                job+=" --model ${model} --precision bf16"
                job+=" --use_auth_token --openai_api_env_keys OPENAI_API_KEY"
                job+=" --tasks ${task} --n_samples 10 --batch_size ${batch_size}"
                job+=" --max_length_generation ${max_length} --temperature 0.8"
                job+=" --allow_code_execution --trust_remote_code --output_dir ${outdir}"
                job+=" --save_generations_raw --save_generations_raw_path ${run_id}_generations_raw.json"
                job+=" --save_generations_prc --save_generations_prc_path ${run_id}_generations_prc.json"
                job+=" --save_references --save_references_path ${run_id}_references.json"
                job+=" --save_results --save_results_path ${run_id}_results.json"
                job+=" &> ${outdir}/${run_id}.log; exit"
                export JOB=${job}; bash SUBMIT.sh
                echo "Submitted ${run_id}"
            done
        done
    done
done
touch ${outdir}/run.done
