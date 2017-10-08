word="[[:blank:]]$1[[:blank:]]"
grep "$word" ../../data/WestburyLab.wikicorp.201004.txt.clean | shuf -n $2
