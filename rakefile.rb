require 'rake'
require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake

define_tasks 'entrec'

task_in_venv :examples do
  vsh 'cd examples/char_rnn && rake'
end

task test: %i[pytest examples]
