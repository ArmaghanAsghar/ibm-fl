#!/bin/zsh

NUM_CLIENTS=10
for ((i=0; i<$NUM_CLIENTS;i++))
do
  echo "starting Party: "$i
  nohup python -m ibmfl.party.party examples/configs/fedavg/keras/config_party$i.yml &
  noh
done
