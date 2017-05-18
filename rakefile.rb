require 'rake'
require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake

PACKAGE = 'simple-entity-recognition'.freeze

define_tasks(PACKAGE, define_pytest: false)

task_in_venv :pytest do
  Dir.glob("#{PACKAGE}/**/*_test.py").each do |file|
    vsh :pytest, file
  end
end

task test: :pytest
