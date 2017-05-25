task :default do
  sh %w[python main.py
        --num_classes 2
        --char_file var/chars.txt
        --train_file var/train/\\*.json
        --eval_file var/eval/\\*.json
        --train_steps 20
        --eval_steps 1
        --batch_size 1
        --batch_queue_capacity 1
        --char_space_size 128].join(' ')
end
