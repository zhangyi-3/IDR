#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
# color
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color


# Set default parameters
p_name=ISPCodec
gpu=4
node=-1
main_file="train.py"


node_prefix=${HOSTNAME%-*}
luster=${node_prefix##*-}
if [ $luster = '15' ]; then
  blind_node=$node_prefix-[22,23,25,26,27,28,29,33,34,35,36,38,39,43,47,53,55,57,58,63]
fi

while [ -n "$1" ]
do
  case "$1" in

    -c)
        p_name=Test;;

    -t)
        p_name=ToolChain;;
    -g)  # gpu number
        gpu=$2
        shift;;
    -n)
        name=$2
        shift;;
    -f)
        main_file=$2
        shift;;
    -p)
        p_name=$2
        shift;;
    *)
        printf "p ${GREEN}$1 is not an option${NC}"
        ;;
  esac
  shift
done


ntask_per_node=$(($gpu<8?$gpu:8))

printf "p ${RED}$p_name${NC}, arun, $arun, gpu ${RED}$gpu${NC}, $ntask_per_node name ${RED}$name${NC}, main_file ${RED}$main_file${NC}\n"

 srun -p $p_name -n$gpu --gres=gpu:$ntask_per_node \
 --ntasks-per-node=$ntask_per_node \
 --cpus-per-task=2 \
 -x $blind_node  \
 -J $name  \
 python -u $main_file -n $name

echo 'EHD----' $(date +%d-%m-%Y" "%H:%M:%S)
