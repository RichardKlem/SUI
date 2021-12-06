#MAKEFLAGS += -j2

all: human

human:
	python3 ./scripts/dicewars-human.py --ai maxn_vector

tournament:
	python3 ./scripts/dicewars-tournament.py --ai-under-test maxn_vector -n 20 -g 4 -r
