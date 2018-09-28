#!/bin/bash

if [[ $USERNAME ]]
then
  ssh $USERNAME@login.leonhard.ethz.ch
else
  echo "set USERNAME: USERNAME=mikhailu connect.sh"
fi
