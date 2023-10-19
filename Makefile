SHELL := /usr/bin/env bash
EXEC = python=3.10
NAME = linc
PACKAGE = eval
ACTIVATE = source activate $(NAME)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin main

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@conda create -yn $(NAME) $(EXEC)
	@$(ACTIVATE) ; \
		python -m pip install -e "." ; \
		conda install -yc conda-forge git-lfs
		
## setup	 : login to huggingface and setup accelerate.
.PHONY : setup
setup : env
	@huggingface-cli login
	@accelerate config

## run       : run the main experiments.
.PHONY : run
run : env outputs/run.done
outputs/run.done : run_expts.sh
	@bash $<

## analyze   : analyze the results.
.PHONY : analyze
analyze : run analysis/analyze.done
analysis/analyze.done : analysis/scripts/analyze_expts.sh
	@cd $(<D) ; bash $(<F)
