task :default do
  sh %W[python main.py
        --num_classes 2
        --char_file var/chars.txt
        --train_file var/train/\*.json
        --eval_file var/eval/\*.json
        --train_steps 20
        --eval_steps 20
        --batch_size 4
        --char_space_size 128].join(' ')
end
