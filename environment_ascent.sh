#!/bin/bash

module reset
module load pgi cuda essl

export PS1="[\u@\h: \w]$ "
alias ls="ls --color"
