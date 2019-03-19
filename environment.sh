#!/bin/bash

module reset
module load pgi
module load cuda

export PS1="[\u@\h: \w]$ "
#export PS1="\[\e[0;90m\][\u@\h: \w]\$\[\e[m\] "
alias ls="ls --color"
