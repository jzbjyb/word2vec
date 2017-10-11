word="[[:blank:]]$1[[:blank:]]"
file=../../data/WestburyLab.wikicorp.201004.txt.clean
#file=../data/text8_lines
grep "$word" "$file" | shuf -n $2
