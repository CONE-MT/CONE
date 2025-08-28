split_start=${1:-21}
split_end=${2:-40}


lgs=$(cat x-x-langs.txt)
lgs=(${lgs//","/ })

mkdir -p spm_logs_update
for i in `seq ${split_start} 1 ${split_end}`;
do
    echo "Start!"
    job_num=$(squeue -u yuanfei | grep "update_" | wc -l)
    if ((${job_num} < 50)); then
      srun -p cpu --gres=gpu:0 -N 1 --ntasks-per-node=1 -J merge_${i}  python -u spm_merge_on_ceph.py cluster3:s3://xujingjing_bucket/opus_all_spm/ \
      cluster3:s3://xujingjing_bucket/opus_all_spm_merge_yf/ ${i} > spm_logs_update/split_${i}_merge.log 2>&1 &
    else
      echo "sleep 120s "
      sleep 120
    fi
done
echo "Finished!"
